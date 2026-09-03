"""OpenCV overlay routines kept separate from monitoring logic."""
from __future__ import annotations
import cv2
from monitoring.models import FrameFeatures


def draw_overlays(frame, features: FrameFeatures, show_landmarks: bool, show_yolo: bool):
    if features.face_bbox:
        x1, y1, x2, y2 = features.face_bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (70,220,100), 2)
    if show_landmarks:
        for x, y in features.landmarks[::4]: cv2.circle(frame, (int(x*frame.shape[1]), int(y*frame.shape[0])), 1, (255,180,0), -1)
    if show_yolo:
        for d in features.object_detections:
            x1,y1,x2,y2 = d.bbox; color = (40,60,240) if d.label == "cell phone" else (255,180,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2); cv2.putText(frame,f"{d.label} {d.confidence:.2f}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    label = f"Faces: {features.face_count} | Gaze: {features.gaze_direction} | Pose: {features.head_direction}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 560), 27), (0,0,0), -1)
    cv2.putText(frame,label,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return frame
