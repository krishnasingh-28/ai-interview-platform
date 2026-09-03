from config import MonitoringConfig
from monitoring.event_engine import EventEngine
from monitoring.models import FrameFeatures, TemporalState


def frame(timestamp, **kwargs):
    result = FrameFeatures(timestamp=timestamp, face_count=1, face_present=True, face_confidence=0.8)
    for key, value in kwargs.items(): setattr(result, key, value)
    return result


def test_event_is_created_once_and_closed_after_recovery():
    config = MonitoringConfig(face_absence_seconds=2.0)
    engine = EventEngine(config)
    active = TemporalState(face_absent_active=True, face_absent_duration=2.1)
    engine.update(frame(2.1, face_present=False, face_count=0), active)
    engine.update(frame(4.0, face_present=False, face_count=0), TemporalState(face_absent_active=True, face_absent_duration=4.0))
    assert len(engine.events) == 1
    event = engine.events[0]
    assert event.start_time == 0.0 and event.duration == 4.0
    engine.update(frame(4.1), TemporalState(face_absent_active=False))
    assert event.end_time == 4.1
    assert "FACE_NOT_VISIBLE" not in engine.open_events


def test_phone_event_requires_persistence():
    config = MonitoringConfig(phone_detection_seconds=1.0)
    engine = EventEngine(config)
    engine.update(frame(0.8, phone_detected=True), TemporalState(phone_active=True, phone_duration=0.8))
    assert not engine.events
    engine.update(frame(1.1, phone_detected=True), TemporalState(phone_active=True, phone_duration=1.1))
    assert [e.event_type for e in engine.events] == ["POSSIBLE_SECONDARY_DEVICE"]
