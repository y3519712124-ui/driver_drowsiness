"""驾驶会话日志和退出摘要。"""
import os
import time
from datetime import datetime

from src.metrics_buffer import MetricsBuffer


class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.start_time = time.time()
        self.end_time = None
        self.max_level = 0
        self.max_score = 0.0
        self.alert_count = 0
        self.voice_turns = 0
        self.frames = 0
        self.face_frames = 0
        self._last_level = None
        self._last_user_text = ""
        self._last_stt_backend = None
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self._log_dir, f"drive_session_{stamp}.log")
        self.summary_path = os.path.join(self._log_dir, f"drive_summary_{stamp}.txt")
        self._write("=== AI 语音副驾驾驶会话 ===")

    def log_start(self, ear_threshold, args):
        self._write(f"启动时间: {self._fmt_time(self.start_time)}")
        self._write(f"EAR阈值: {ear_threshold:.3f}")
        self._write(f"语音启用: {not args.no_voice}")
        self._write(f"离线语音识别: {getattr(args, 'offline_stt', False)}")
        self._write(f"在线语音识别: {getattr(args, 'online_stt', False)}")
        self._write(f"LLM启用: {not args.no_llm}")
        self._write(f"校准模式: {args.calibrate}")

    def update_frame(self, metrics, level, score):
        self.frames += 1
        if metrics.face_detected:
            self.face_frames += 1
        if score > self.max_score:
            self.max_score = score
        if level > self.max_level:
            self.max_level = level
        if level != self._last_level:
            self._last_level = level
            self._write(f"疲劳等级变化: level={level}, score={score:.1f}")

    def record_alert(self, duration):
        self.alert_count += 1
        self._write(f"硬报警触发: 闭眼 {duration:.1f}s")

    def update_voice(self, user_text, reply_text):
        if user_text and user_text != self._last_user_text:
            self.voice_turns += 1
            self._last_user_text = user_text
            self._write(f"司机语音: {user_text}")
            if reply_text:
                self._write(f"副驾回复: {reply_text}")

    def update_stt_backend(self, backend):
        if backend != self._last_stt_backend:
            self._last_stt_backend = backend
            self._write(f"语音识别后端: {backend}")

    def build_summary(self, snapshot):
        self.end_time = time.time()
        duration = max(0.0, self.end_time - self.start_time)
        face_ratio = (self.face_frames / self.frames * 100.0) if self.frames else 0.0
        microsleeps = MetricsBuffer.count_microsleeps(snapshot)
        yawns = MetricsBuffer.count_yawns(snapshot)
        perclos = MetricsBuffer.compute_perclos(snapshot)
        blink_rate = MetricsBuffer.compute_blink_rate(snapshot)

        lines = [
            "AI 语音副驾驾驶摘要",
            f"开始时间: {self._fmt_time(self.start_time)}",
            f"结束时间: {self._fmt_time(self.end_time)}",
            f"运行时长: {duration / 60.0:.1f} 分钟",
            f"最高疲劳等级: {self.max_level}",
            f"最高疲劳评分: {self.max_score:.1f}",
            f"硬报警次数: {self.alert_count}",
            f"语音交互次数: {self.voice_turns}",
            f"微睡眠次数: {microsleeps}",
            f"打哈欠次数: {yawns}",
            f"PERCLOS: {perclos * 100:.1f}%",
            f"眨眼频率: {blink_rate:.1f} 次/分钟",
            f"人脸可见率: {face_ratio:.1f}%",
        ]
        return "\n".join(lines)

    def finish(self, snapshot):
        summary = self.build_summary(snapshot)
        self._write("")
        self._write(summary)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        return summary

    def _write(self, text):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    @staticmethod
    def _fmt_time(ts):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
