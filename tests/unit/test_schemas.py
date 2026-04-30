"""Unit tests for Pydantic schema validation."""

from datetime import datetime

from src.storage.schemas import (
    EarthquakeEvent,
    FireDetection,
    GdeltEvent,
    IngestionRun,
    IngestionStatus,
    PortMetric,
    WeatherAlert,
)


def test_weather_alert_severity_normalisation():
    alert = WeatherAlert(
        alert_id="test-001",
        port_id="la_lb",
        event_type="High Wind Warning",
        severity="  SEVERE  ",        # should be normalised to "Severe"
        certainty="Likely",
        urgency="Expected",
        headline="High winds expected",
        description="Sustained winds 45-55 mph",
        area_description="Los Angeles County Coast",
        effective=datetime.utcnow(),
    )
    assert alert.severity == "Severe"


def test_weather_alert_expires_optional():
    alert = WeatherAlert(
        alert_id="test-002",
        port_id="ny_nj",
        event_type="Fog Advisory",
        severity="Minor",
        certainty="Likely",
        urgency="Future",
        headline="Dense fog advisory",
        description="Visibility below 1/4 mile",
        area_description="New York Harbor",
        effective=datetime.utcnow(),
        expires=None,
    )
    assert alert.expires is None


def test_gdelt_event_fields():
    evt = GdeltEvent(
        global_event_id=123456789,
        sql_date="20240415",
        event_code="145",
        event_root_code="14",
        goldstein_scale=-5.0,
        num_mentions=12,
        num_articles=4,
        avg_tone=-3.2,
        nearest_port_id="la_lb",
        distance_to_port_km=45.3,
        source_url="https://example.com/news",
    )
    assert evt.event_root_code == "14"
    assert evt.goldstein_scale == -5.0


def test_port_metric_optional_fields():
    m = PortMetric(
        port_id="houston",
        port_name="Houston",
        week_ending="2024-04-08",
        dataset_id="y7iz-hhid",
    )
    assert m.median_berthing_time_hrs is None
    assert m.teu_count is None


def test_earthquake_event():
    eq = EarthquakeEvent(
        usgs_id="us7000test",
        magnitude=5.2,
        mag_type="mw",
        depth_km=12.3,
        lat=33.9,
        lon=-118.5,
        place="20 km SW of Los Angeles",
        occurred_at=datetime.utcnow(),
        nearest_port_id="la_lb",
        distance_to_port_km=22.1,
        tsunami_flag=False,
    )
    assert eq.magnitude == 5.2
    assert not eq.tsunami_flag


def test_fire_detection_confidence_normalisation():
    fire = FireDetection(
        lat=34.1,
        lon=-118.3,
        brightness=350.0,
        confidence="N",         # should normalise to "nominal"
        satellite="NOAA-20",
        acquired_at=datetime.utcnow(),
        nearest_port_id="la_lb",
        distance_to_port_km=80.0,
    )
    assert fire.confidence == "nominal"


def test_ingestion_run_defaults():
    run = IngestionRun(
        run_id="abc123",
        source="nws",
        started_at=datetime.utcnow(),
    )
    assert run.status == IngestionStatus.SUCCESS
    assert run.records_fetched == 0
    assert run.bronze_paths == []
