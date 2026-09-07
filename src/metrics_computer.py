"""纯函数：面部指标计算。"""

import math
import numpy as np

import config


def landmark_to_pixel(lm, w, h):
    """将归一化关键点转为像素坐标。"""
    return np.array([lm.x * w, lm.y * h])


def compute_ear(landmarks, w, h, upper_ids, lower_ids, left_ids, right_ids):
    """计算单眼 Eye Aspect Ratio。"""
    upper = landmark_to_pixel(landmarks[upper_ids[0]], w, h)
    lower = landmark_to_pixel(landmarks[lower_ids[0]], w, h)
    left  = landmark_to_pixel(landmarks[left_ids[0]],  w, h)
    right = landmark_to_pixel(landmarks[right_ids[0]], w, h)

    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def compute_both_ear(landmarks, w, h):
    """计算双眼平均 EAR。"""
    left_ear = compute_ear(landmarks, w, h,
                           config.LEFT_EYE_UPPER, config.LEFT_EYE_LOWER,
                           config.LEFT_EYE_LEFT, config.LEFT_EYE_RIGHT)
    right_ear = compute_ear(landmarks, w, h,
                            config.RIGHT_EYE_UPPER, config.RIGHT_EYE_LOWER,
                            config.RIGHT_EYE_LEFT, config.RIGHT_EYE_RIGHT)
    return left_ear, right_ear, (left_ear + right_ear) / 2.0


def compute_mar(landmarks, w, h):
    """计算 Mouth Aspect Ratio（打哈欠检测）。"""
    upper = landmark_to_pixel(landmarks[config.MOUTH_UPPER[0]], w, h)
    lower = landmark_to_pixel(landmarks[config.MOUTH_LOWER[0]], w, h)
    left  = landmark_to_pixel(landmarks[config.MOUTH_LEFT[0]],  w, h)
    right = landmark_to_pixel(landmarks[config.MOUTH_RIGHT[0]], w, h)

    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_head_pose(landmarks, w, h):
    """估算头部 pitch 角（点头检测）。
    使用鼻尖与两眼内角中点的垂直偏移量。
    """
    left_inner = landmark_to_pixel(landmarks[133], w, h)
    right_inner = landmark_to_pixel(landmarks[362], w, h)
    nose = landmark_to_pixel(landmarks[config.NOSE_TIP], w, h)

    eye_mid = (left_inner + right_inner) / 2.0
    eye_distance = np.linalg.norm(left_inner - right_inner)
    if eye_distance == 0:
        return 0.0, 0.0

    delta_y = nose[1] - eye_mid[1]
    pitch_rad = math.atan2(delta_y, eye_distance)
    pitch_deg = math.degrees(pitch_rad)

    return pitch_deg, eye_distance


def get_blendshape_scores(blendshapes):
    """提取 eyeBlink 和 jawOpen 的 blendshape 分数。"""
    scores = {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, "jawOpen": 0.0}
    if not blendshapes:
        return scores
    for bs in blendshapes:
        name = bs.category_name
        if name in scores:
            scores[name] = bs.score
    return scores
