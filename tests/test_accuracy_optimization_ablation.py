import pytest
import numpy as np
from src.models.accuracy_optimization_ablation import (
    ExperimentRecord,
    ExperimentRunner,
    DomainOptimizationEngine,
    VehicleOptimizationPipeline,
    PedestrianOptimizationPipeline,
)


def test_experiment_runner_single_run():
    runner = ExperimentRunner(seed=42)
    w_tr = np.random.randn(20, 200, 6).astype(np.float32)
    t_tr = np.random.randn(20, 2).astype(np.float32)
    w_va = np.random.randn(5, 200, 6).astype(np.float32)
    t_va = np.random.randn(5, 2).astype(np.float32)
    w_te = np.random.randn(5, 200, 6).astype(np.float32)
    t_te = np.random.randn(5, 2).astype(np.float32)

    cfg = {
        "architecture": "TCNVelocityEstimator",
        "learning_rate": 0.001,
        "epochs": 2,
        "batch_size": 8,
        "capacity_channels": 32,
        "dropout": 0.1,
    }

    rec = runner.train_single_run(
        config=cfg,
        train_windows=w_tr,
        train_targets=t_tr,
        val_windows=w_va,
        val_targets=t_va,
        test_windows=w_te,
        test_targets=t_te,
        experiment_id="EXP_TEST_001",
        domain="vehicle",
        dataset_name="IO-VNBD",
    )

    assert isinstance(rec, ExperimentRecord)
    assert rec.experiment_id == "EXP_TEST_001"
    assert rec.domain == "vehicle"
    assert rec.dataset_name == "IO-VNBD"
    assert rec.random_seed == 42
    assert "val_mae_mps" in rec.validation_metrics
    assert "test_mae_mps" in rec.final_test_metrics
    assert rec.param_count > 0
    assert rec.inference_latency_ms >= 0.0


def test_17_dimension_ablation_engine():
    engine = DomainOptimizationEngine(domain="vehicle")
    records, best_rec = engine.run_17_dimension_ablation(dataset_name="Navigators India Vehicle")

    assert len(records) == 18
    assert isinstance(best_rec, ExperimentRecord)

    exp_ids = [r.experiment_id for r in records]
    assert len(set(exp_ids)) == 18

    dimensions = [r.configuration.get("dim") for r in records]
    assert "baseline" in dimensions
    assert "sequence_length_100" in dimensions
    assert "architecture_resnet1d" in dimensions
    assert "architecture_lstm" in dimensions
    assert "optimizer_adamw" in dimensions
    assert "loss_function_smoothl1" in dimensions
    assert "augmentation_enabled" in dimensions

    best_val_score = min(
        records,
        key=lambda r: r.validation_metrics["unseen_session_mae_mps"] + 0.01 * r.inference_latency_ms
    )
    assert best_rec.experiment_id == best_val_score.experiment_id


def test_vehicle_optimization_pipeline():
    pipeline = VehicleOptimizationPipeline()
    res = pipeline.run_full_vehicle_optimization()

    assert res["domain"] == "vehicle"
    assert "IO-VNBD" in res["dataset_results"]
    assert "Navigators India Vehicle" in res["dataset_results"]
    assert "IO-VNBD + Navigators India Combined" in res["dataset_results"]
    assert "IO-VNBD Pretrained + Indian Fine-tuned" in res["dataset_results"]
    assert res["total_experiments"] == 72


def test_pedestrian_optimization_pipeline():
    pipeline = PedestrianOptimizationPipeline()
    res = pipeline.run_full_pedestrian_optimization()

    assert res["domain"] == "pedestrian"
    assert "RoNIN" in res["dataset_results"]
    assert "OxIOD" in res["dataset_results"]
    assert "Navigators India Pedestrian" in res["dataset_results"]
    assert "RoNIN + OxIOD + Indian Combined" in res["dataset_results"]
    assert res["total_experiments"] == 72
