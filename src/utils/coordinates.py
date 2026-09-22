"""
Navigators IDR - Coordinate Transforms
WGS84 (LLA) ↔ ENU (East-North-Up) conversions, bearing, and distance.

References:
    - WGS84 ellipsoid parameters from NIMA TR8350.2
    - ENU conversion from Bowring (1985)
"""

import numpy as np
from typing import Tuple


# --- WGS84 Ellipsoid Constants ---
WGS84_A = 6378137.0               # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563     # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F) # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # First eccentricity squared


def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / np.pi


def _prime_vertical_radius(lat_rad: float) -> float:
    """
    Compute the radius of curvature in the prime vertical (N).

    Args:
        lat_rad: Geodetic latitude in radians.

    Returns:
        Prime vertical radius of curvature in meters.
    """
    sin_lat = np.sin(lat_rad)
    return WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
    """
    Convert geodetic coordinates (LLA) to Earth-Centered Earth-Fixed (ECEF).

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude above WGS84 ellipsoid in meters.

    Returns:
        np.ndarray: [X, Y, Z] in meters (ECEF frame).
    """
    lat_r = deg2rad(lat)
    lon_r = deg2rad(lon)
    N = _prime_vertical_radius(lat_r)

    X = (N + alt) * np.cos(lat_r) * np.cos(lon_r)
    Y = (N + alt) * np.cos(lat_r) * np.sin(lon_r)
    Z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_r)

    return np.array([X, Y, Z])


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic (LLA) using Bowring's iterative method.

    Args:
        x, y, z: ECEF coordinates in meters.

    Returns:
        Tuple of (latitude_deg, longitude_deg, altitude_m).
    """
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)

    # Initial estimate
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    for _ in range(10):  # Converges in ~3 iterations
        N = _prime_vertical_radius(lat)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    N = _prime_vertical_radius(lat)
    alt = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-10 else abs(z) / abs(np.sin(lat)) - N * (1 - WGS84_E2)

    return rad2deg(lat), rad2deg(lon), alt


def lla_to_enu(
    lat: float, lon: float, alt: float,
    ref_lat: float, ref_lon: float, ref_alt: float = 0.0
) -> np.ndarray:
    """
    Convert LLA to local East-North-Up (ENU) coordinates relative to a reference point.

    Args:
        lat, lon, alt: Target point in degrees / meters.
        ref_lat, ref_lon, ref_alt: Reference origin in degrees / meters.

    Returns:
        np.ndarray: [East, North, Up] in meters relative to reference.
    """
    # Convert both to ECEF
    target_ecef = lla_to_ecef(lat, lon, alt)
    ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    delta = target_ecef - ref_ecef

    # Rotation matrix from ECEF to ENU
    ref_lat_r = deg2rad(ref_lat)
    ref_lon_r = deg2rad(ref_lon)

    sin_lat = np.sin(ref_lat_r)
    cos_lat = np.cos(ref_lat_r)
    sin_lon = np.sin(ref_lon_r)
    cos_lon = np.cos(ref_lon_r)

    R = np.array([
        [-sin_lon,              cos_lon,             0.0     ],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,   cos_lat ],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,   sin_lat ]
    ])

    return R @ delta


def enu_to_lla(
    east: float, north: float, up: float,
    ref_lat: float, ref_lon: float, ref_alt: float = 0.0
) -> Tuple[float, float, float]:
    """
    Convert local ENU coordinates back to LLA.

    Args:
        east, north, up: ENU coordinates in meters.
        ref_lat, ref_lon, ref_alt: Reference origin in degrees / meters.

    Returns:
        Tuple of (latitude_deg, longitude_deg, altitude_m).
    """
    ref_lat_r = deg2rad(ref_lat)
    ref_lon_r = deg2rad(ref_lon)

    sin_lat = np.sin(ref_lat_r)
    cos_lat = np.cos(ref_lat_r)
    sin_lon = np.sin(ref_lon_r)
    cos_lon = np.cos(ref_lon_r)

    # Inverse rotation (transpose of ENU→ECEF rotation)
    R_inv = np.array([
        [-sin_lon, -sin_lat * cos_lon,  cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon,  cos_lat * sin_lon],
        [ 0.0,      cos_lat,            sin_lat           ]
    ])

    delta_ecef = R_inv @ np.array([east, north, up])
    ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    target_ecef = ref_ecef + delta_ecef

    return ecef_to_lla(target_ecef[0], target_ecef[1], target_ecef[2])


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Compute the great-circle distance between two LLA points using the Haversine formula.

    Args:
        lat1, lon1: First point in degrees.
        lat2, lon2: Second point in degrees.

    Returns:
        Distance in meters.
    """
    lat1_r, lon1_r = deg2rad(lat1), deg2rad(lon1)
    lat2_r, lon2_r = deg2rad(lat2), deg2rad(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return WGS84_A * c


def compute_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Compute the initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: Start point in degrees.
        lat2, lon2: End point in degrees.

    Returns:
        Bearing in radians [0, 2π), measured clockwise from North.
    """
    lat1_r, lon1_r = deg2rad(lat1), deg2rad(lon1)
    lat2_r, lon2_r = deg2rad(lat2), deg2rad(lon2)

    dlon = lon2_r - lon1_r

    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)

    bearing = np.arctan2(x, y)
    return bearing % (2 * np.pi)


def destination_point(
    lat: float, lon: float,
    bearing: float, distance: float
) -> Tuple[float, float]:
    """
    Compute the destination point given a start, bearing, and distance.

    Args:
        lat, lon: Start point in degrees.
        bearing: Bearing in radians.
        distance: Distance in meters.

    Returns:
        Tuple of (latitude_deg, longitude_deg) of destination.
    """
    lat_r = deg2rad(lat)
    lon_r = deg2rad(lon)
    angular_dist = distance / WGS84_A

    dest_lat = np.arcsin(
        np.sin(lat_r) * np.cos(angular_dist)
        + np.cos(lat_r) * np.sin(angular_dist) * np.cos(bearing)
    )
    dest_lon = lon_r + np.arctan2(
        np.sin(bearing) * np.sin(angular_dist) * np.cos(lat_r),
        np.cos(angular_dist) - np.sin(lat_r) * np.sin(dest_lat)
    )

    return rad2deg(dest_lat), rad2deg(dest_lon)
