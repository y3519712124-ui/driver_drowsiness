import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass

import config


@dataclass
class DetectionResult:
    face_detected: bool
    landmarks: list | None        # MediaPipe NormalizedLandmark list
    blendshapes: list | None      # MediaPipe Classifications list
    image_width: int
    image_height: int


class FaceDetector:
    def __init__(self):
        self._ensure_model()
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_model():
        if not os.path.exists(config.MODEL_PATH):
            print("正在下载 face_landmarker 模型...")
            urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
            print("模型下载完成！")

    def detect(self, bgr_frame) -> DetectionResult:
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if results.face_landmarks:
            return DetectionResult(
                face_detected=True,
                landmarks=results.face_landmarks[0],
                blendshapes=results.face_blendshapes[0] if results.face_blendshapes else None,
                image_width=w,
                image_height=h,
            )
        return DetectionResult(face_detected=False, landmarks=None, blendshapes=None, image_width=w, image_height=h)
