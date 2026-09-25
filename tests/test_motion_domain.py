import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.motion_domain import (
    MotionCategory,
    MotionClassificationResult,
    RuleBasedMotionClassifier,
    VehicleModel,
    PedestrianModel,
    SharedNavigationFusion,
)
from navigation.interfaces import AIVelocityMeasurement


def test_motion_classifier_vehicle_and_pedestrian():
    classifier = RuleBasedMotionClassifier()

    vehicle_imu = np.random.normal(loc=[0.1, 0.0, 9.81, 0.0, 0.0, 0.0], scale=0.05, size=(50, 6))
    res_veh = classifier.classify_motion(vehicle_imu)
    assert res_veh.category == MotionCategory.VEHICLE
    assert res_veh.confidence > 0.5

    ped_imu = np.random.normal(loc=[0.0, 0.0, 9.81, 0.0, 0.0, 0.0], scale=2.5, size=(50, 6))
    res_ped = classifier.classify_motion(ped_imu)
    assert res_ped.category == MotionCategory.PEDESTRIAN


def test_pedestrian_model_estimation():
    model = PedestrianModel()
    imu_window = np.random.normal(loc=[0.5, 0.2, 9.81, 0.0, 0.0, 0.0], scale=1.5, size=(30, 6))
    meas = model.estimate_velocity(imu_window, dt=0.1, heading=0.0)

    assert meas is not None
    assert isinstance(meas, AIVelocityMeasurement)
    assert meas.is_valid
    assert meas.variance_north > 0.0


def test_shared_navigation_fusion_routing():
    classifier = RuleBasedMotionClassifier()
    ped_model = PedestrianModel()
    veh_model = VehicleModel(underlying_engine=None)
    fusion = SharedNavigationFusion(classifier=classifier, vehicle_model=veh_model, pedestrian_model=ped_model)

    ped_imu = np.random.normal(loc=[0.0, 0.0, 9.81, 0.0, 0.0, 0.0], scale=2.5, size=(50, 6))
    class_res, vel_meas = fusion.process_imu_window(ped_imu, dt=0.1)

    assert class_res.category == MotionCategory.PEDESTRIAN
    assert fusion.active_mode == MotionCategory.PEDESTRIAN
    assert vel_meas is not None

    veh_imu = np.random.normal(loc=[0.0, 0.0, 9.81, 0.0, 0.0, 0.0], scale=0.05, size=(50, 6))
    class_res_v, vel_meas_v = fusion.process_imu_window(veh_imu, dt=0.1)

    assert class_res_v.category == MotionCategory.VEHICLE
    assert fusion.active_mode == MotionCategory.VEHICLE
    assert vel_meas_v is None
