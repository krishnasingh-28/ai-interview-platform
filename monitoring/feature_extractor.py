"""Transforms raw detector results into one frame-level monitoring record."""
from __future__ import annotations
from config import PHONE_CONFIDENCE
from monitoring.models import FrameFeatures


def build_features(timestamp, face, pose, gaze, detections, quality) -> FrameFeatures:
    """Build a resilient FrameFeatures value from optional vision outputs."""
    phone = any(d.label == "cell phone" and d.confidence >= PHONE_CONFIDENCE for d in detections)
    persons = sum(d.label == "person" for d in detections)
    return FrameFeatures(timestamp=timestamp, face_count=face.face_count, face_present=face.face_count > 0,
        face_confidence=face.confidence, yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll,
        head_direction=pose.direction, gaze_direction=gaze.direction, gaze_confidence=gaze.confidence,
        phone_detected=phone, person_count=persons, face_bbox=face.bbox, landmarks=face.landmarks,
        object_detections=detections, frame_quality=quality)
