"""Unit tests for config helpers — no I/O, no external calls."""

import pytest

from config import PORT_BY_ID, US_PORTS, haversine_km, nearest_port


def test_haversine_known_distance():
    # LA to New York is approximately 3940 km
    dist = haversine_km(34.05, -118.24, 40.71, -74.01)
    assert 3900 < dist < 4000, f"Unexpected distance: {dist}"


def test_haversine_zero():
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_nearest_port_la():
    # Coordinates right next to the Port of Los Angeles
    port, dist = nearest_port(33.73, -118.27)
    assert port.id == "la_lb"
    assert dist < 10, f"Distance too large: {dist}"


def test_nearest_port_ny():
    port, dist = nearest_port(40.70, -74.04)
    assert port.id == "ny_nj"


def test_all_ports_have_required_fields():
    for port in US_PORTS:
        assert port.id, f"Port missing id: {port}"
        assert port.name, f"Port missing name: {port}"
        assert -90 <= port.lat <= 90, f"Invalid lat for {port.id}: {port.lat}"
        assert -180 <= port.lon <= 180, f"Invalid lon for {port.id}: {port.lon}"


def test_port_by_id_lookup():
    assert PORT_BY_ID["la_lb"].name == "Los Angeles / Long Beach"
    assert PORT_BY_ID["houston"].state == "TX"
    assert "savannah" in PORT_BY_ID
