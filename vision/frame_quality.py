"""Simple, explainable image-quality signals."""
from __future__ import annotations
import cv2
import numpy as np
from config import (MAX_BRIGHTNESS, MIN_BLUR_VARIANCE, MIN_BRIGHTNESS, MIN_FACE_AREA_RATIO,
                    MIN_FACE_CONFIDENCE, QUALITY_REASON_PENALTY)
from monitoring.models import FrameQuality


def assess_frame_quality(frame: np.ndarray, face_bbox: tuple[int, int, int, int] | None, face_confidence: float) -> FrameQuality:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness, blur = float(np.mean(gray)), float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS: reasons.append("extreme_brightness")
    if blur < MIN_BLUR_VARIANCE: reasons.append("blur")
    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        if ((x2-x1)*(y2-y1)) / (frame.shape[0]*frame.shape[1]) < MIN_FACE_AREA_RATIO: reasons.append("small_face")
    if face_bbox and face_confidence < MIN_FACE_CONFIDENCE: reasons.append("low_face_quality")
    # Missing face is tracked separately, avoiding a duplicate obstruction event.
    score = max(0.0, 1.0 - QUALITY_REASON_PENALTY * len(reasons))
    return FrameQuality(score, brightness, blur, bool(reasons), reasons)
