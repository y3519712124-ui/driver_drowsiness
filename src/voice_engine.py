"""后台语音引擎 — Windows SAPI / pyttsx3 回退。"""
import threading
import queue
import time


class VoiceEngine:
    def __init__(self, rate=0, volume=100):
        self._queue = queue.Queue(maxsize=1)  # 只保留最新一条消息
        self._thread = None
        self._running = False
        self._rate = rate
        self._volume = volume
        self._error_count = 0
        self._is_speaking = False
        self._current_text = ""
        self._last_spoken_text = ""
        self._last_spoken_ended_at = 0.0

    @property
    def running(self):
        return self._running

    @property
    def speaking(self):
        return self._is_speaking

    @property
    def current_text(self):
        return self._current_text

    @property
    def last_spoken_text(self):
        return self._last_spoken_text

    @property
    def last_spoken_ended_at(self):
        return self._last_spoken_ended_at

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VoiceEngine")
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._put_nowait(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def speak(self, text: str):
        """发送语音。如果上一条还没播完，直接替换为新消息。"""
        if not self._running:
            return
        # 清空旧消息，放入新消息
        self._put_nowait(text)

    def _put_nowait(self, text):
        """安全放入队列：先清空再放新的。"""
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _loop(self):
        voice = self._init_voice()
        if voice is None:
            print("[VoiceEngine] 语音初始化失败")
            return

        print("[VoiceEngine] 语音引擎就绪")

        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                print(f"[Voice] {text[:40]}...")
                self._is_speaking = True
                self._current_text = text
                self._speak_text(voice, text)
                self._error_count = 0
            except Exception as e:
                self._error_count += 1
                print(f"[VoiceEngine] 播放失败 (#{self._error_count}): {e}")
                if self._error_count > 5:
                    print("[VoiceEngine] 连续失败过多，尝试重新初始化...")
                    voice = self._init_voice()
                    self._error_count = 0
            finally:
                self._is_speaking = False
                self._last_spoken_text = text
                self._last_spoken_ended_at = time.time()
                self._current_text = ""

            self._queue.task_done()

    def _init_voice(self):
        # SAPI
        try:
            import win32com.client
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            if self._rate:
                voice.Rate = self._rate
            voice.Volume = self._volume
            desc = voice.Voice.GetDescription()
            print(f"[VoiceEngine] SAPI: {desc}")
            return ("sapi", voice)
        except ImportError:
            pass
        except Exception as e:
            print(f"[VoiceEngine] SAPI 失败: {e}")

        # pyttsx3 回退
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.setProperty("volume", 1.0)
            print("[VoiceEngine] pyttsx3 就绪")
            return ("pyttsx3", engine)
        except ImportError:
            pass
        except Exception as e:
            print(f"[VoiceEngine] pyttsx3 失败: {e}")

        return None

    def _speak_text(self, voice, text):
        backend, obj = voice
        if backend == "sapi":
            obj.Speak(text, 0)  # 同步
        else:
            obj.say(text)
            obj.runAndWait()
