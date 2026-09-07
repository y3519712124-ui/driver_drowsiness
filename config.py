import os

# === 摄像头 ===
CAMERA_INDEX = 0

# === 模型路径 ===
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'face_landmarker.task')
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# === 眼部关键点索引（MediaPipe 478点模型）===
LEFT_EYE_UPPER  = [159]
LEFT_EYE_LOWER  = [145]
LEFT_EYE_LEFT   = [33]
LEFT_EYE_RIGHT  = [133]

RIGHT_EYE_UPPER = [386]
RIGHT_EYE_LOWER = [374]
RIGHT_EYE_LEFT  = [362]
RIGHT_EYE_RIGHT = [263]

# === 嘴部关键点索引（打哈欠检测）===
MOUTH_UPPER = [13]
MOUTH_LOWER = [14]
MOUTH_LEFT  = [61]
MOUTH_RIGHT = [291]

# === 头部姿态关键点 ===
NOSE_TIP = 1

# === 绘制用关键点 ===
FACE_CONTOUR_IDS = list(range(0, 468, 3))
LEFT_EYE_DRAW_IDS  = [33, 159, 145, 23, 24, 133, 7, 163, 144, 188, 222]
RIGHT_EYE_DRAW_IDS = [362, 386, 374, 257, 263, 398, 384, 385, 387, 388, 466]

# === EAR 阈值 ===
EAR_THRESHOLD = 0.21        # EAR 低于此值判定为闭眼
EYE_CLOSED_THRESHOLD = 0.18 # 更严格的闭眼阈值，用于眨眼检测
CONSEC_FRAMES = 2           # 连续 N 帧闭眼才认定（防抖）
WAIT_TIME = 2.0             # 微睡眠超此时长触发硬报警

# === MAR（打哈欠）阈值 ===
YAWN_MAR_THRESHOLD = 0.5    # MAR 超此值判定为张嘴
YAWN_JAW_OPEN_THRESHOLD = 0.45  # jawOpen blendshape 超此值也判定为张嘴
YAWN_DURATION = 1.0         # 张嘴超此时长（秒）判定为打哈欠

# === 眨眼分类阈值 ===
BLINK_NORMAL_MAX = 0.2      # 正常眨眼 < 200ms
BLINK_LONG_MAX = 0.5        # 长眨眼 200-500ms
# >500ms 且 < 2s 判定为微睡眠

# === MediaPipe 配置 ===
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# === 疲劳分析窗口 ===
SHORT_WINDOW_SEC = 2.0      # 短期窗口（眨眼事件检测）
MEDIUM_WINDOW_SEC = 30.0    # 中期窗口（眨眼频率、打哈欠）
LONG_WINDOW_SEC = 120.0     # 长期窗口（EAR 趋势）

# === 疲劳评分权重 ===
WEIGHT_BLINK_RATE = 0.18
WEIGHT_MICROSLEEP = 0.30
WEIGHT_BLINK_DURATION = 0.15
WEIGHT_HEAD_NOD = 0.15
WEIGHT_YAWN = 0.10
WEIGHT_EAR_DRIFT = 0.05
WEIGHT_PERCLOS = 0.07

# === 疲劳等级阈值（带迟滞）===
FATIGUE_LEVELS = [
    {"level": 0, "enter": 15, "exit": 999, "label": "清醒", "cooldown": 999},
    {"level": 1, "enter": 20, "exit": 12,  "label": "轻度", "cooldown": 120},
    {"level": 2, "enter": 35, "exit": 25,  "label": "中度", "cooldown": 60},
    {"level": 3, "enter": 55, "exit": 45,  "label": "重度", "cooldown": 25},
    {"level": 4, "enter": 75, "exit": 65,  "label": "危险", "cooldown": 5},
]

# === 副驾对话 ===
LLM_TIMEOUT = 12.0          # 用户主动提问时的 LLM 超时秒数
LLM_PROACTIVE_ENABLED = False  # 主动疲劳提醒默认使用模板，避免 7B 模型拖慢主循环
TEMPLATE_HISTORY_SIZE = 10  # 最近不重复的模板数
FACE_LOSS_TIMEOUT = 10.0    # 无人脸超时，暂停说话

# === LLM 模型 ===
LLM_MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# === 语音识别 (STT) ===
import os as _os
VOSK_MODEL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "models", "vosk-model-small-cn-0.22")
STT_SAMPLE_RATE = 16000          # Vosk 标准采样率
STT_FRAMES_PER_BUFFER = 1600    # 每帧 100ms
STT_REPLY_COOLDOWN = 2.0       # 两次自动回复的最小间隔，防止连续触发
STT_MIN_AUDIO_LEVEL = 0.006     # 低于此音量的识别结果视为噪声
STT_TTS_MUTE_TAIL = 1.2        # TTS 结束后继续静音麦克风，防止识别尾音
STT_ASSISTANT_ECHO_WINDOW = 8.0 # 副驾播报结束后，短时间过滤疑似回声文本
