"""Conservative iris-position gaze approximation for demo visualisation."""
from __future__ import annotations
from monitoring.models import GazeResult
from config import (GAZE_CONFIDENCE_BASE, GAZE_CONFIDENCE_CAP, GAZE_CONFIDENCE_SCALE,
                    GAZE_HORIZONTAL_THRESHOLD, GAZE_MIN_EYE_HEIGHT, GAZE_MIN_EYE_WIDTH,
                    GAZE_VERTICAL_THRESHOLD)

# MediaPipe 478-landmark iris indices; older 468-landmark models safely return UNKNOWN.
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_VERTICAL = (159, 145)
RIGHT_EYE_VERTICAL = (386, 374)


def estimate_gaze(landmarks: list[tuple[float, float]]) -> GazeResult:
    """Estimate relative iris displacement, returning UNKNOWN for insufficient data."""
    if len(landmarks) < 478:
        return GazeResult()
    try:
        iris_x = sum(landmarks[i][0] for i in LEFT_IRIS + RIGHT_IRIS) / 10
        iris_y = sum(landmarks[i][1] for i in LEFT_IRIS + RIGHT_IRIS) / 10
        cx = sum(landmarks[i][0] for i in LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS) / 4
        cy = sum(landmarks[i][1] for i in LEFT_EYE_VERTICAL + RIGHT_EYE_VERTICAL) / 4
        eye_width = abs(landmarks[33][0] - landmarks[133][0]) + abs(landmarks[362][0] - landmarks[263][0])
        eye_height = abs(landmarks[159][1] - landmarks[145][1]) + abs(landmarks[386][1] - landmarks[374][1])
        if eye_width < GAZE_MIN_EYE_WIDTH or eye_height < GAZE_MIN_EYE_HEIGHT:
            return GazeResult()
        dx, dy = (iris_x - cx) / eye_width, (iris_y - cy) / eye_height
        confidence = min(GAZE_CONFIDENCE_CAP, GAZE_CONFIDENCE_BASE + eye_width * GAZE_CONFIDENCE_SCALE)
        if abs(dx) > abs(dy) and abs(dx) > GAZE_HORIZONTAL_THRESHOLD:
            return GazeResult("RIGHT" if dx > 0 else "LEFT", confidence)
        if abs(dy) > GAZE_VERTICAL_THRESHOLD:
            return GazeResult("DOWN" if dy > 0 else "UP", confidence)
        return GazeResult("CENTER", confidence)
    except (IndexError, ZeroDivisionError):
        return GazeResult()
