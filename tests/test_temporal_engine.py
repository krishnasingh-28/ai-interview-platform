from config import MonitoringConfig
from monitoring.models import FrameFeatures
from monitoring.temporal_engine import TemporalEngine


def feature(timestamp: float, **changes) -> FrameFeatures:
    base = FrameFeatures(timestamp=timestamp, face_count=1, face_present=True)
    for key, value in changes.items():
        setattr(base, key, value)
    return base


def test_face_absence_accumulates_then_resets():
    engine = TemporalEngine(MonitoringConfig())
    assert engine.update(feature(0, face_present=False, face_count=0)).face_absent_duration == 0
    assert engine.update(feature(1.2, face_present=False, face_count=0)).face_absent_duration == 1.2
    assert engine.update(feature(2.1, face_present=False, face_count=0)).face_absent_duration == 2.1
    assert engine.update(feature(2.2)).face_absent_duration == 0


def test_multiple_face_head_gaze_and_phone_persistence():
    engine = TemporalEngine(MonitoringConfig())
    engine.update(feature(0, face_count=2))
    state = engine.update(feature(1.1, face_count=2, yaw=35, gaze_direction="LEFT", phone_detected=True))
    assert state.multiple_face_duration == 1.1
    # Separate continuous signals begin at their first true frame, not a prior unrelated frame.
    state = engine.update(feature(4.2, face_count=2, yaw=35, gaze_direction="LEFT", phone_detected=True))
    assert state.head_turn_duration == 3.1
    assert state.gaze_deviation_duration == 3.1
    assert state.phone_duration == 3.1
