"""Tests for Phone-to-Vehicle Alignment and Coordinate Frame Transformations."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.alignment import (
    PhoneVehicleAligner,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)


class TestAlignmentMath:
    def test_quaternion_rotation_matrix_roundtrip(self):

        angles = [
            (0.0, 0.0, 0.0),
            (0.2, -0.3, 0.5),
            (np.pi / 4, -np.pi / 6, np.pi / 3),
            (-0.1, 0.4, -0.8),
        ]
        for roll, pitch, yaw in angles:
            R = euler_to_rotation_matrix(roll, pitch, yaw)
            q = rotation_matrix_to_quaternion(R)
            R_recovered = quaternion_to_rotation_matrix(q)
            assert np.allclose(R, R_recovered, atol=1e-6)

            r_rec, p_rec, y_rec = rotation_matrix_to_euler(R_recovered)
            assert np.allclose([roll, pitch, yaw], [r_rec, p_rec, y_rec], atol=1e-5)

    def test_orthogonality(self):
        R = euler_to_rotation_matrix(0.3, -0.5, 0.8)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


class TestPhoneVehicleAligner:
    def test_flat_phone_alignment(self):
        """Phone flat on dashboard, facing vehicle front: phone frame = vehicle frame."""
        aligner = PhoneVehicleAligner()

        static_accel = np.array([[0.0, 0.0, 9.81]])
        forward_hint = np.array([1.0, 0.0, 0.0])

        R = aligner.compute_alignment_matrix(static_accel, forward_hint=forward_hint)
        assert np.allclose(R, np.eye(3), atol=1e-3)


        a_phone = np.array([2.0, 0.0, 9.81])
        a_veh = aligner.transform_accel(a_phone)
        assert np.allclose(a_veh, a_phone, atol=1e-3)

    def test_tilted_phone_alignment(self):
        """Phone mounted in landscape or tilted at 45 degrees pitch."""
        aligner = PhoneVehicleAligner()
        pitch_angle = np.pi / 4



        R_true = euler_to_rotation_matrix(0.0, pitch_angle, 0.0)

        g_veh = np.array([0.0, 0.0, 9.81])
        g_phone = R_true.T @ g_veh

        aligner.compute_alignment_matrix(g_phone, forward_hint=R_true.T @ np.array([1.0, 0.0, 0.0]))
        R_est = aligner.get_rotation_matrix()


        g_transformed = aligner.transform_accel(g_phone)
        assert np.allclose(g_transformed / np.linalg.norm(g_transformed), np.array([0.0, 0.0, 1.0]), atol=1e-3)

    def test_batch_transformation(self):
        aligner = PhoneVehicleAligner()
        aligner.set_orientation(0.1, -0.2, 0.4)
        N = 50
        accel_data = np.random.randn(N, 3)
        accel_veh = aligner.transform_accel(accel_data)
        assert accel_veh.shape == (N, 3)


        for i in range(5):
            single = aligner.transform_accel(accel_data[i])
            assert np.allclose(single, accel_veh[i])
