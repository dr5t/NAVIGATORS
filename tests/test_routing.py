"""
Tests for Phase 31: Routing Engine Integration
Verifies:
1. RoutingEngine computes recommended, alternative, and offline route structures.
2. Turn-by-turn steps, distance, and duration calculations.
3. Offline mode correctly flags offline routes.
4. FastAPI calculate_route_endpoint returns valid response payload.
"""

import pytest
from src.navigation.routing import RoutingEngine, Route, RouteStep
from src.api.routing import calculate_route_endpoint, CalculateRouteRequest


def test_routing_engine_compute_route():
    """Verify RoutingEngine calculates recommended, alternative, and offline routes."""
    engine = RoutingEngine()

    origin = (28.6139, 77.2090)
    destination = (28.7041, 77.1025)

    res = engine.compute_route(origin, destination, is_offline=False)

    assert res["status"] == "success"
    assert res["is_offline"] is False


    rec = res["recommended"]
    assert rec is not None
    assert rec["type"] == "recommended"
    assert rec["total_distance_meters"] > 0.0
    assert len(rec["waypoints"]) >= 2
    assert len(rec["steps"]) >= 1


    alt = res["alternative"]
    assert alt is not None
    assert alt["type"] == "alternative"
    assert alt["total_distance_meters"] > rec["total_distance_meters"]


    off = res["offline"]
    assert off is not None
    assert off["type"] == "offline"


def test_routing_engine_offline_mode():
    """In offline mode, alternative online options are omitted."""
    engine = RoutingEngine()
    origin = (30.3165, 78.0322)
    destination = (30.3250, 78.0410)

    res = engine.compute_route(origin, destination, is_offline=True)

    assert res["is_offline"] is True
    assert res["alternative"] is None
    assert res["offline"] is not None


def test_routing_api_endpoint():
    """Verify calculate_route_endpoint API response."""
    req = CalculateRouteRequest(
        origin=(28.6139, 77.2090),
        destination=(28.6200, 77.2150),
        is_offline=False,
    )

    resp = calculate_route_endpoint(req=req)
    assert resp["status"] == "success"
    assert "recommended" in resp
    assert "steps" in resp["recommended"]
