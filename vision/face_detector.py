"""MediaPipe Tasks Face Landmarker adapter with safe no-model fallbacks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from config import (FACE_CONFIDENCE_AREA_NORMALIZER, FACE_MODEL_DETECTION_CONFIDENCE,
                    FACE_MODEL_PRESENCE_CONFIDENCE, FACE_MODEL_TRACKING_CONFIDENCE, MAX_FACES)


@dataclass
class FaceDetectionResult:
    face_count: int
    confidence: float
    bbox: tuple[int, int, int, int] | None
    landmarks: list[tuple[float, float]]
    all_landmarks: list[list[tuple[float, float]]]


class FaceDetector:
    """Runs MediaPipe Face Landmarker in VIDEO mode when an asset is available."""
    def __init__(self, model_path: str) -> None:
        self.landmarker = None
        self.error: str | None = None
        if not Path(model_path).exists():
            self.error = f"Face Landmarker asset not found: {model_path}"
            return
        try:
            import mediapipe as mp
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=MAX_FACES,
                min_face_detection_confidence=FACE_MODEL_DETECTION_CONFIDENCE,
                min_face_presence_confidence=FACE_MODEL_PRESENCE_CONFIDENCE,
                min_tracking_confidence=FACE_MODEL_TRACKING_CONFIDENCE,
                output_face_blendshapes=False,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as exc:  # model/version errors should not stop the UI
            self.error = str(exc)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceDetectionResult:
        """Return normalized landmark results; return an empty result on failures."""
        if self.landmarker is None:
            return FaceDetectionResult(0, 0.0, None, [], [])
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            result = self.landmarker.detect_for_video(image, timestamp_ms)
            groups = [[(p.x, p.y) for p in face] for face in result.face_landmarks]
            if not groups:
                return FaceDetectionResult(0, 0.0, None, [], [])
            h, w = frame_bgr.shape[:2]
            first = groups[0]
            xs, ys = [p[0] for p in first], [p[1] for p in first]
            x1, y1 = max(0, int(min(xs) * w)), max(0, int(min(ys) * h))
            x2, y2 = min(w - 1, int(max(xs) * w)), min(h - 1, int(max(ys) * h))
            # Tasks landmarks have no meaningful detector score; use visible-area quality.
            confidence = min(1.0, max(0.0, ((x2-x1)*(y2-y1)) / (w*h*FACE_CONFIDENCE_AREA_NORMALIZER)))
            return FaceDetectionResult(len(groups), confidence, (x1, y1, x2, y2), first, groups)
        except Exception:
            return FaceDetectionResult(0, 0.0, None, [], [])

    def close(self) -> None:
        if self.landmarker:
            self.landmarker.close()
