from monitoring.feature_extractor import build_features
from monitoring.models import FrameQuality, GazeResult, HeadPoseResult, ObjectDetection
from vision.face_detector import FaceDetectionResult


def test_feature_extractor_combines_detector_outputs():
    face = FaceDetectionResult(1, .8, (1, 2, 20, 30), [(0.1, .2)], [[(.1, .2)]])
    detection = ObjectDetection("cell phone", .9, (3, 4, 30, 40))
    quality = FrameQuality(.9, 100, 120, False)
    result = build_features(4.0, face, HeadPoseResult(yaw=8, direction="CENTER"), GazeResult("CENTER", .5), [detection], quality)
    assert result.face_present and result.phone_detected
    assert result.face_bbox == (1, 2, 20, 30)
    assert result.yaw == 8 and result.gaze_direction == "CENTER"
