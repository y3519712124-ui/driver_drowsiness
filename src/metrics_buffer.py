"""线程安全的指标缓冲区 + 滑动窗口统计。"""
import time
import threading
from collections import deque
from dataclasses import dataclass, field

import config


@dataclass
class FatigueFrameMetrics:
    timestamp: float
    face_detected: bool

    # 眼部
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    eye_blink_left: float = 0.0
    eye_blink_right: float = 0.0

    # 嘴部
    mar: float = 0.0
    jaw_open: float = 0.0

    # 头部姿态
    head_pitch: float = 0.0

    # 判断结果
    eyes_closed: bool = False           # 当前帧是否判定为闭眼
    blink_event: bool = False           # 是否刚完成一次眨眼（上升沿）


@dataclass
class BlinkRecord:
    """一次眨眼的记录。"""
    timestamp: float
    duration: float                     # 秒
    category: str                       # "normal" | "long" | "microsleep"


@dataclass
class YawnRecord:
    """一次打哈欠的记录。"""
    timestamp: float
    duration: float


@dataclass
class FatigueAnalysis:
    """疲劳分析结果。"""
    level: int
    score: float
    sub_scores: dict = field(default_factory=dict)
    factors: list = field(default_factory=list)


class MetricsBuffer:
    """线程安全的指标缓冲区，维护滑动窗口并提供统计计算。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._short_frames = deque()     # 短期帧数据（2秒）
        self._frame_history = deque()    # 中期帧数据（PERCLOS）
        self._blink_records = deque()    # 眨眼记录
        self._yawn_records = deque()     # 打哈欠记录
        self._ear_history = deque()      # EAR 长期记录

        # 闭眼帧状态追踪（跨帧）
        self._closed_start_time = None
        self._was_closed = False

        # 张嘴状态追踪
        self._yawn_start_time = None
        self._was_yawning = False

        # 头部点头检测
        self._pitch_history = deque(maxlen=90)  # ~3秒

    def push(self, metrics: FatigueFrameMetrics):
        """推入一帧指标。"""
        with self._lock:
            now = time.time()
            self._short_frames.append(metrics)
            self._frame_history.append(metrics)
            self._pitch_history.append((now, metrics.head_pitch))
            self._ear_history.append((now, metrics.avg_ear))

            # 清理过期数据
            self._prune(now)

            # 检测眨眼完成事件
            if self._was_closed and not metrics.eyes_closed and self._closed_start_time is not None:
                # 闭眼结束 → 记录一次眨眼
                duration = now - self._closed_start_time
                cat = self._classify_blink(duration)
                self._blink_records.append(BlinkRecord(timestamp=now, duration=duration, category=cat))
                self._closed_start_time = None

            if metrics.eyes_closed and self._closed_start_time is None:
                self._closed_start_time = now

            self._was_closed = metrics.eyes_closed

            # 打哈欠检测：MAR 和 MediaPipe jawOpen 任一满足即可，提高鲁棒性。
            is_yawning = (
                metrics.mar > config.YAWN_MAR_THRESHOLD
                or metrics.jaw_open > config.YAWN_JAW_OPEN_THRESHOLD
            )
            if self._was_yawning and not is_yawning and self._yawn_start_time is not None:
                duration = now - self._yawn_start_time
                if duration >= config.YAWN_DURATION:
                    self._yawn_records.append(YawnRecord(timestamp=now, duration=duration))
                self._yawn_start_time = None

            if is_yawning and self._yawn_start_time is None:
                self._yawn_start_time = now

            self._was_yawning = is_yawning

    def get_snapshot(self):
        """获取当前缓冲区快照（线程安全）。"""
        with self._lock:
            now = time.time()
            return {
                "short_frames": list(self._short_frames),
                "frame_history": list(self._frame_history),
                "blink_records": list(self._blink_records),
                "yawn_records": list(self._yawn_records),
                "ear_history": list(self._ear_history),
                "pitch_history": list(self._pitch_history),
                "snapshot_time": now,
            }

    def _prune(self, now):
        """清理过期数据。"""
        while self._short_frames and now - self._short_frames[0].timestamp > config.SHORT_WINDOW_SEC:
            self._short_frames.popleft()
        while self._frame_history and now - self._frame_history[0].timestamp > config.MEDIUM_WINDOW_SEC:
            self._frame_history.popleft()
        while self._blink_records and now - self._blink_records[0].timestamp > config.MEDIUM_WINDOW_SEC:
            self._blink_records.popleft()
        while self._yawn_records and now - self._yawn_records[0].timestamp > config.LONG_WINDOW_SEC:
            self._yawn_records.popleft()
        while self._ear_history and now - self._ear_history[0][0] > config.LONG_WINDOW_SEC:
            self._ear_history.popleft()

    @staticmethod
    def _classify_blink(duration):
        if duration < config.BLINK_NORMAL_MAX:
            return "normal"
        elif duration < config.BLINK_LONG_MAX:
            return "long"
        else:
            return "microsleep"

    # === 统计方法（调用方需要持锁，这里不加锁，由 get_snapshot 保证线程安全）===

    @staticmethod
    def compute_blink_rate(snapshot):
        """计算眨眼频率（次/分钟）。"""
        now = snapshot["snapshot_time"]
        cutoff = now - config.MEDIUM_WINDOW_SEC
        recent_blinks = [b for b in snapshot["blink_records"] if b.timestamp > cutoff]
        if not recent_blinks:
            # 如果 30s 内没有眨眼，可能是刚启动，返回基准值
            return 12.0
        count = len(recent_blinks)
        span = now - min(b.timestamp for b in recent_blinks)
        if span <= 0:
            return 12.0
        return (count / span) * 60.0

    @staticmethod
    def compute_avg_blink_duration(snapshot):
        """计算平均眨眼时长（秒）。"""
        now = snapshot["snapshot_time"]
        cutoff = now - config.MEDIUM_WINDOW_SEC
        recent = [b for b in snapshot["blink_records"] if b.timestamp > cutoff]
        if not recent:
            return 0.15  # 默认正常眨眼时长 ~150ms
        return sum(b.duration for b in recent) / len(recent)

    @staticmethod
    def count_microsleeps(snapshot):
        """统计 60 秒内的微睡眠次数。"""
        now = snapshot["snapshot_time"]
        cutoff = now - 60
        return sum(1 for b in snapshot["blink_records"]
                   if b.timestamp > cutoff and b.category == "microsleep")

    @staticmethod
    def count_yawns(snapshot):
        """统计 120 秒内的打哈欠次数。"""
        now = snapshot["snapshot_time"]
        cutoff = now - config.LONG_WINDOW_SEC
        return sum(1 for y in snapshot["yawn_records"] if y.timestamp > cutoff)

    @staticmethod
    def compute_perclos(snapshot):
        """计算中期窗口内闭眼帧占比（PERCLOS）。"""
        now = snapshot["snapshot_time"]
        cutoff = now - config.MEDIUM_WINDOW_SEC
        frames = [f for f in snapshot.get("frame_history", []) if f.timestamp > cutoff and f.face_detected]
        if len(frames) < 5:
            return 0.0
        closed = sum(1 for f in frames if f.eyes_closed)
        return closed / len(frames)

    @staticmethod
    def compute_head_nod_intensity(snapshot):
        """计算头部点头强度（pitch 方差）。"""
        pitches = [p for _, p in snapshot["pitch_history"]]
        if len(pitches) < 10:
            return 0.0
        mean = sum(pitches) / len(pitches)
        variance = sum((p - mean) ** 2 for p in pitches) / len(pitches)
        return variance

    @staticmethod
    def compute_ear_drift(snapshot):
        """计算 EAR 基线漂移（长期均值相对初始值下降百分比）。"""
        ear_history = snapshot["ear_history"]
        if len(ear_history) < 30:
            return 0.0
        # 取前 10% 作为初始基线
        first_n = max(10, len(ear_history) // 10)
        first_vals = [v for _, v in ear_history[:first_n] if v > 0.1]  # 排除异常低值
        if not first_vals:
            return 0.0
        baseline = sum(first_vals) / len(first_vals)
        # 取最近一段的均值
        last_n = max(10, len(ear_history) // 5)
        last_vals = [v for _, v in ear_history[-last_n:] if v > 0.1]
        if not last_vals:
            return 0.0
        current = sum(last_vals) / len(last_vals)
        if baseline <= 0:
            return 0.0
        drift = (baseline - current) / baseline  # 正值表示下降
        return max(0.0, drift)  # 只关心下降
