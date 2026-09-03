"""Head-pose calculation from stable Face Landmarker points via solvePnP."""
from __future__ import annotations
import cv2
import numpy as np
from monitoring.models import HeadPoseResult

LANDMARK_IDS = (1, 152, 33, 263, 61, 291)
MODEL_POINTS = np.array([(0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26),
                         (43.3, 32.7, -26), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)


def estimate_head_pose(landmarks: list[tuple[float, float]], shape: tuple[int, ...], yaw_limit: float, pitch_limit: float) -> HeadPoseResult:
    """Use a generic 3D face model; angles are approximate and intended for a demo."""
    if len(landmarks) <= max(LANDMARK_IDS):
        return HeadPoseResult()
    try:
        h, w = shape[:2]
        image_points = np.array([(landmarks[i][0] * w, landmarks[i][1] * h) for i in LANDMARK_IDS], dtype=np.float64)
        camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rotation, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return HeadPoseResult()
        rmat, _ = cv2.Rodrigues(rotation)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = (float(v) for v in angles)
        direction = "CENTER"
        if abs(yaw) > abs(pitch) and abs(yaw) > yaw_limit: direction = "RIGHT" if yaw > 0 else "LEFT"
        elif abs(pitch) > pitch_limit: direction = "DOWN" if pitch > 0 else "UP"
        return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll, direction=direction, confidence=0.6)
    except (cv2.error, ValueError, IndexError):
        return HeadPoseResult()
