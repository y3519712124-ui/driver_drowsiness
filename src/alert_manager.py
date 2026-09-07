"""硬报警管理 — 仅疲劳等级 4（危险）时触发。"""
import time
import threading
import winsound


class AlertManager:
    def __init__(self):
        self._last_beep = 0
        self._beep_interval = 0.7  # 蜂鸣间隔
        self._beeping = False

    def trigger(self):
        """非阻塞播放一次硬报警（蜂鸣）。调用方可循环调用直到危险解除。"""
        now = time.time()
        if self._beeping or now - self._last_beep < self._beep_interval:
            return False

        self._last_beep = now
        self._beeping = True
        threading.Thread(target=self._beep_once, daemon=True, name="AlertBeep").start()
        return True

    def _beep_once(self):
        try:
            winsound.Beep(880, 300)
        finally:
            self._beeping = False

    def reset(self):
        self._last_beep = 0
        self._beeping = False
