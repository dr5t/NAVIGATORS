"""Macro summaries and paired cluster bootstrap; windows are not independent trials."""
from collections import defaultdict
import numpy as np

from data.india_dataset.schema import digest


def cohort(result, region):
    source = result["input_identity"]
    return dict(source_dataset=source["source_dataset"], source_kind=source["source_kind"],
                release=source["manifest_sha256"], reference_kind=source["reference_kind"],
                reference_source_dataset=source["reference_source_dataset"],
                scenario=result["case"]["kind"], duration_s=result["case"]["duration_s"],
                phase=region["phase"], environment=region["environment"])


def aggregate(results):
    groups = {}
    for result in results:
        if result["status"] != "completed":
            continue
        for region in result["regions"]:
            group = {**cohort(result, region), "architecture": result["architecture"]}
            key = digest(group)
            entry = groups.setdefault(key, dict(cohort=group, values=defaultdict(lambda: defaultdict(list)), jobs=set()))
            entry["jobs"].add(result["job_id"])
            for metric, value in region["metrics"].items():
                entry["values"][metric]  # Keep unavailable metrics explicit in aggregate output.
                if value is not None:
                    entry["values"][metric][result["input_identity"]["session_id"]].append(value)
    output = []
    for key, group in sorted(groups.items()):
        metrics = {}
        for metric, sessions in sorted(group["values"].items()):
            values = [float(np.mean(v)) for v in sessions.values()]
            metrics[metric] = dict(mean=float(np.mean(values)) if values else None, std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                                   sessions=len(values), session_means={s: float(np.mean(v)) for s,v in sorted(sessions.items())})
        output.append(dict(group_id=key, cohort=group["cohort"], metrics=metrics, job_ids=sorted(group["jobs"])))
    return output


def paired_statistics(results, config):
    """Compare F vs production, A–E, and each single-removal system on matched cases.

    Pairing includes reference support indices. Starts/repeats are averaged within
    release split_group before resampling. Positive delta means F is worse for
    error/latency; the material-improvement decision is limited to position RMSE.
    """
    complete = {r["job_id"]: r for r in results if r["status"] == "completed"}
    by_case = defaultdict(dict)
    for result in complete.values():
        key = digest([result["input_identity"], result["case"], result["repeat"]])
        by_case[key][result["architecture"]] = result
    groups = {}
    for case_results in by_case.values():
        full = case_results.get("F")
        if full is None:
            continue
        for name, comparator in case_results.items():
            if name == "F":
                continue
            counterparts = {(r["phase"], r["environment"]): r for r in comparator["regions"]}
            for region in full["regions"]:
                other = counterparts[(region["phase"], region["environment"])]
                group = {**cohort(full, region), "candidate": "F", "comparator": name}
                for metric, a in region["metrics"].items():
                    b = other["metrics"].get(metric)
                    if a is None or b is None or region["support"][metric]["sample_indices"] != other["support"][metric]["sample_indices"]:
                        continue
                    group_metric = {**group, "metric": metric}
                    entry = groups.setdefault(digest(group_metric), dict(comparison=group_metric, clusters=defaultdict(list), pairs=[]))
                    cluster = full["input_identity"]["split_group"]
                    entry["clusters"][cluster].append(float(a-b))
                    entry["pairs"].append(dict(full_job=full["job_id"], comparator_job=comparator["job_id"],
                                               delta=float(a-b), full_value=a, comparator_value=b,
                                               support_sha256=digest(region["support"][metric]["sample_indices"])))
    output = []
    for key, entry in sorted(groups.items()):
        deltas = np.array([np.mean(v) for _,v in sorted(entry["clusters"].items())])
        n = len(deltas)
        rng = np.random.default_rng(config["seed"] + int(key[:8],16))
        ci, p = None, None
        if n >= 2:
            boot = np.array([np.mean(rng.choice(deltas, n, replace=True)) for _ in range(config["bootstrap_samples"])])
            ci = np.percentile(boot, [2.5,97.5]).tolist()
            observed = abs(float(np.mean(deltas)))
            if n <= 14:
                null = [abs(float(np.mean(deltas * np.array([1 if mask & (1<<i) else -1 for i in range(n)])))) for mask in range(1<<n)]
                p = sum(value >= observed-1e-12 for value in null)/len(null)
            else:
                null = [abs(float(np.mean(deltas*rng.choice([-1,1], n)))) for _ in range(config["bootstrap_samples"])]
                p = (1+sum(value >= observed-1e-12 for value in null))/(len(null)+1)
        output.append(dict(comparison_id=key, **entry["comparison"], independent_groups=n,
                           paired_cases=len(entry["pairs"]), mean_delta=float(np.mean(deltas)),
                           paired_cluster_bootstrap_ci95=ci, paired_sign_randomization_p=p,
                           holm_adjusted_p=None, evidence="insufficient independent groups" if n<2 else "inconclusive",
                           pairs=entry["pairs"]))
    # One correction family across all reported comparisons, metrics and strata.
    eligible = sorted([r for r in output if r["paired_sign_randomization_p"] is not None], key=lambda r:r["paired_sign_randomization_p"])
    previous = 0.
    for i, row in enumerate(eligible):
        previous = min(1., max(previous, (len(eligible)-i)*row["paired_sign_randomization_p"]))
        row["holm_adjusted_p"] = previous
        if (row["metric"] == "position_rmse_m" and previous < .05 and
                row["paired_cluster_bootstrap_ci95"][1] < -config["material_improvement_m"]):
            row["evidence"] = "material reduction in position RMSE under this declared protocol"
    return output
