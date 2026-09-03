"""Ultralytics YOLO adapter, restricted to persons and cell phones."""
from __future__ import annotations
from pathlib import Path
from config import YOLO_PERSON_CLASS, YOLO_PHONE_CLASS
from monitoring.models import ObjectDetection


class ObjectDetector:
    def __init__(self, model_name: str) -> None:
        self.model = None
        self.error: str | None = None
        if not Path(model_name).exists():
            self.error = f"YOLO model asset not found: {model_name}"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except Exception as exc:
            self.error = str(exc)

    def detect(self, frame) -> list[ObjectDetection]:
        if self.model is None:
            return []
        try:
            output = self.model(frame, verbose=False, classes=[YOLO_PERSON_CLASS, YOLO_PHONE_CLASS])[0]
            names = output.names
            detections = []
            for box in output.boxes:
                label = names[int(box.cls[0])]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(ObjectDetection(label, float(box.conf[0]), (x1, y1, x2, y2)))
            return detections
        except Exception:
            return []
