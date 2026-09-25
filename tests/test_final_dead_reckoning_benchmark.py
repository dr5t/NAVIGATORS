import pytest
from pathlib import Path
from src.benchmarks.final_dead_reckoning_benchmark import (
    BaselineType,
    StatisticalDistribution,
    OutageEvaluationResult,
    BenchmarkSuiteResult,
    BenchmarkEvaluator,
    FinalDeadReckoningBenchmarkSuite,
)


def test_statistical_distribution_computation():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    dist = StatisticalDistribution.from_list(vals)

    assert dist.mean == 3.0
    assert dist.median == 3.0
    assert dist.std > 0.0
    assert dist.p95 > 4.0
    assert dist.per_session_values == vals


def test_benchmark_evaluator_synthetic_session_and_outage():
    evaluator = BenchmarkEvaluator(seed=42)
    session = evaluator.generate_synthetic_session(domain="vehicle", num_samples=500)

    assert session["domain"] == "vehicle"
    assert len(session["t"]) == 500
    assert "true_vn" in session

    res = evaluator.simulate_baseline_outage(
        session=session,
        baseline=BaselineType.FULL_NAVIGATORS,
        outage_dur_sec=60,
    )

    assert isinstance(res, OutageEvaluationResult)
    assert res.domain == "vehicle"
    assert res.baseline_type == BaselineType.FULL_NAVIGATORS.value
    assert res.outage_duration_sec == 60
    assert res.gnss_anomaly_precision > 0.90
    assert res.recovery_time_sec < 1.0


def test_final_benchmark_suite_vehicle_and_pedestrian():
    suite = FinalDeadReckoningBenchmarkSuite(seed=42)

    veh_res = suite.run_vehicle_benchmark()
    assert isinstance(veh_res, BenchmarkSuiteResult)
    assert veh_res.domain == "vehicle"
    assert len(veh_res.outage_results) == 25
    assert "urban_dense" in veh_res.domain_stratified_breakdown
    assert len(veh_res.failure_cases) > 0
    assert veh_res.reproducibility_metadata["test_set_touched"] is False
    assert veh_res.reproducibility_metadata["model_auto_promoted"] is False

    ped_res = suite.run_pedestrian_benchmark()
    assert isinstance(ped_res, BenchmarkSuiteResult)
    assert ped_res.domain == "pedestrian"
    assert len(ped_res.outage_results) == 25
    assert "handheld" in ped_res.domain_stratified_breakdown
    assert len(ped_res.failure_cases) > 0


def test_export_benchmark_artifacts(tmp_path):
    suite = FinalDeadReckoningBenchmarkSuite(seed=42)
    veh_res = suite.run_vehicle_benchmark()
    ped_res = suite.run_pedestrian_benchmark()

    json_file, csv_file = suite.export_benchmark_artifacts(
        vehicle_res=veh_res,
        pedestrian_res=ped_res,
        output_dir=str(tmp_path),
    )

    assert json_file.exists()
    assert csv_file.exists()
    assert json_file.stat().st_size > 0
    assert csv_file.stat().st_size > 0
