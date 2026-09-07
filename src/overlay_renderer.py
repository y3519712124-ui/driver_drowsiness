"""画面叠加渲染（使用 PIL 支持中文显示）。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


def _get_font(size=24):
    """获取中文字体，优先级：微软雅黑 > 宋体 > 默认。"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑 粗体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _cv2_add_chinese_text(img, text, position, color, size=24, bold=False):
    """使用 PIL 在 OpenCV 图像上绘制中文文字。"""
    if not text:
        return

    font = _get_font(size)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 转换为 PIL 用的颜色 (RGB, no alpha)
    pil_color = (color[2], color[1], color[0])

    draw.text(position, text, font=font, fill=pil_color)

    # 转回 OpenCV 格式
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[:, :, :] = result[:, :, :]


def _fit_text(text, max_chars=28):
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


class OverlayRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_face_landmarks(self, frame, landmarks, w, h):
        for idx in config.FACE_CONTOUR_IDS:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

    def draw_eye_landmarks(self, frame, landmarks, w, h):
        for idx in config.LEFT_EYE_DRAW_IDS:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 200, 0), -1)
        for idx in config.RIGHT_EYE_DRAW_IDS:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 200, 0), -1)

    def draw_status(self, frame, text, y=30, color=(0, 255, 0)):
        _cv2_add_chinese_text(frame, f'状态: {text}', (10, y), color, size=22)

    def draw_ear(self, frame, avg_ear, eyes_detected, y=60):
        color = (0, 255, 0) if eyes_detected else (0, 0, 255)
        _cv2_add_chinese_text(frame, f'EAR: {avg_ear:.2f}', (10, y), color, size=22)

    def draw_closed_timer(self, frame, duration, y=90):
        _cv2_add_chinese_text(frame, f'闭眼: {duration:.1f}s', (10, y), (0, 0, 255), size=22)

    def draw_alert(self, frame, text="疲劳警报！", y=150):
        _cv2_add_chinese_text(frame, text, (50, y), (0, 0, 255), size=36)

    def draw_blendshapes(self, frame, blendshapes, y=120):
        if not blendshapes:
            return
        offset = 0
        for bs in blendshapes:
            if 'eyeBlink' in bs.category_name:
                cv2.putText(frame, f'{bs.category_name}: {bs.score:.2f}',
                            (10, y + offset), self.font, 0.4, (200, 200, 200), 1)
                offset += 20

    def draw_fatigue_level(self, frame, level, score, y=180):
        level_colors = {
            0: (0, 255, 0),
            1: (0, 255, 255),
            2: (0, 165, 255),
            3: (0, 0, 255),
            4: (0, 0, 255),
        }
        level_labels = {0: "清醒", 1: "轻度疲劳", 2: "中度疲劳", 3: "重度疲劳", 4: "危险！"}
        color = level_colors.get(level, (255, 255, 255))
        label = level_labels.get(level, "?")
        _cv2_add_chinese_text(frame, f'疲劳: {label} ({score:.0f})', (10, y), color, size=22)

    def draw_no_face(self, frame, y=30):
        _cv2_add_chinese_text(frame, "未检测到人脸", (10, y), (0, 165, 255), size=22)

    def draw_co_pilot_status(self, frame, state, y=210):
        state_labels = {
            "IDLE": "待命", "OBSERVING": "观察中",
            "ENGAGING": "互动中", "WARNING": "警告",
            "CRITICAL": "紧急",
        }
        label = state_labels.get(state, state)
        color = (180, 180, 180)
        _cv2_add_chinese_text(frame, f'副驾: {label}', (10, y), color, size=18)

    def draw_mic_indicator(self, frame, active, audio_level=0.0, backend="", x=None, y=30):
        """麦克风状态指示灯 + 音频电平条。"""
        if x is None:
            x = frame.shape[1] - 40
        color = (0, 255, 0) if active else (80, 80, 80)
        cv2.circle(frame, (x, y), 8, color, -1)
        _cv2_add_chinese_text(frame, "MIC", (x - 15, y + 12), color, size=12)
        if backend:
            _cv2_add_chinese_text(frame, backend.upper(), (x - 30, y + 28), color, size=11)

        # 音频电平条
        bar_x = x + 15
        bar_h = int(40 * min(audio_level * 20, 1.0))  # 放大20倍
        if bar_h > 0:
            bar_color = (0, 255, 0) if audio_level < 0.3 else (0, 255, 255)
            cv2.rectangle(frame, (bar_x, y + bar_h), (bar_x + 6, y), bar_color, -1)
        cv2.rectangle(frame, (bar_x, y + 40), (bar_x + 6, y), (60, 60, 60), 1)

    def draw_stt_hint(self, frame, y=270):
        """提示用户始终监听中。"""
        _cv2_add_chinese_text(frame, '语音副驾始终在线，直接说话即可', (10, y), (120, 120, 120), size=14)

    def draw_calibration(self, frame, remaining, sample_count, message="请自然睁眼看向前方"):
        """绘制 EAR 校准提示。"""
        text = f"校准中 {remaining:.1f}s  样本:{sample_count}"
        _cv2_add_chinese_text(frame, message, (40, 130), (0, 255, 255), size=28)
        _cv2_add_chinese_text(frame, text, (40, 170), (0, 255, 255), size=22)

    def draw_conversation(self, frame, user_text="", assistant_text="", y=None):
        """显示最近一句司机语音和副驾回复。"""
        if y is None:
            y = frame.shape[0] - 58
        if user_text:
            _cv2_add_chinese_text(
                frame, f"司机: {_fit_text(user_text)}", (10, y), (220, 220, 220), size=16
            )
        if assistant_text:
            _cv2_add_chinese_text(
                frame, f"副驾: {_fit_text(assistant_text)}", (10, y + 24), (120, 220, 255), size=16
            )
