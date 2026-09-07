"""语音识别引擎 — Google STT 为主（高准确率），Vosk 离线为回退。
始终监听模式：任何识别到的语音自动触发回复，无需唤醒词或结束语。"""
import threading
import time
import numpy as np
import re
from difflib import SequenceMatcher

import config

WAKE_ONLY_PATTERNS = {
    "副驾", "副驾副驾", "副家", "副家副家", "付驾", "付驾付驾",
    "小副驾", "小副驾小副驾", "副驾驶", "副驾驶副驾驶",
}
NOISE_FRAGMENTS = {"B这首歌", "逼这首歌", "比这首歌", "这首歌", "副驾副驾"}
FILLER_CHARS = set("的了啊呀嗯额呃哦吧吗嘛呢")
ASSISTANT_ECHO_MIN_LEN = 4
INTENT_HINTS = (
    "打开音乐", "放音乐", "播放", "放歌", "听歌", "来首歌", "来一首歌", "关闭",
    "关掉", "停止", "暂停", "继续", "状态", "疲劳", "困", "什么时候", "为什么",
    "怎么办", "怎么", "哪里", "导航", "天气", "笑话",
)


class STTEngine:
    def __init__(self, model_path=None, backend="auto"):
        self._running = False
        self._thread = None
        self._model_path = model_path or config.VOSK_MODEL_PATH
        self._preferred_backend = backend

        self._recognizer = None
        self._microphone = None
        self._vosk_recognizer = None
        self._vosk_audio = None
        self._vosk_stream = None
        self._backend = None

        self._on_user_speech = None
        self._muted = False
        self._audio_level = 0.0
        self._last_reply_time = 0
        self._last_text = ""
        self._last_text_time = 0
        self._mute_until = 0.0
        self._assistant_text = ""
        self._assistant_text_time = 0.0

    # === 回调 ===
    def on_user_speech(self, callback):
        self._on_user_speech = callback

    def set_muted(self, muted: bool):
        self._muted = muted
        if muted:
            self._mute_until = time.time() + config.STT_TTS_MUTE_TAIL

    def set_assistant_text(self, text: str, ended_at: float = None):
        """记录副驾刚播报过的文字，用于过滤扬声器回声。"""
        normalized = self._normalize_text(text)
        if not normalized:
            return
        ended_at = ended_at or time.time()
        if normalized == self._assistant_text and ended_at <= self._assistant_text_time:
            return
        self._assistant_text = normalized
        self._assistant_text_time = ended_at

    # === 属性 ===
    @property
    def audio_level(self):
        return self._audio_level

    @property
    def backend(self):
        return self._backend or self._preferred_backend

    # === 启动 / 停止 ===
    def start(self):
        if self._running:
            return
        if not self._init_backend():
            print("[STT] 语音识别初始化失败，语音输入不可用")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="STTEngine")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._vosk_stream:
            try:
                self._vosk_stream.stop_stream()
                self._vosk_stream.close()
            except Exception:
                pass
            self._vosk_stream = None
        if self._vosk_audio:
            try:
                self._vosk_audio.terminate()
            except Exception:
                pass
            self._vosk_audio = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # === 内部 ===
    def _find_mic_index(self, sr_module):
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            default_info = p.get_default_input_device_info()
            default_name = default_info.get('name', '')
            p.terminate()
            for i, name in enumerate(sr_module.Microphone.list_microphone_names()):
                if default_name and default_name in name:
                    return i
        except Exception:
            pass
        return None

    def _init_backend(self):
        if self._preferred_backend == "vosk":
            return self._init_vosk()

        # 1) Google STT
        if self._preferred_backend in ("auto", "google"):
            try:
                import speech_recognition as sr
                self._recognizer = sr.Recognizer()
                self._recognizer.energy_threshold = 300
                self._recognizer.dynamic_energy_threshold = True
                self._recognizer.pause_threshold = 0.8

                mic_index = self._find_mic_index(sr)
                if mic_index is not None:
                    self._microphone = sr.Microphone(device_index=mic_index)
                    print(f"[STT] 使用麦克风 [{mic_index}]")
                else:
                    self._microphone = sr.Microphone()

                with self._microphone as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self._backend = "google"
                print("[STT] 使用 Google 在线识别 (中文) — 始终监听模式")
                return True
            except ImportError:
                print("[STT] speech_recognition 未安装")
            except Exception as e:
                print(f"[STT] speech_recognition 初始化失败: {e}")

        # 2) Vosk 回退
        if self._init_vosk():
            return True

        return False

    def _switch_to_vosk(self, reason):
        print(f"[STT] {reason}，切换到 Vosk 离线识别...")
        if self._init_vosk():
            self._backend = "vosk"
            self._loop_vosk()
            return True
        print("[STT] Vosk 切换失败，继续尝试当前后端")
        return False

    @staticmethod
    def _is_network_error(error):
        text = str(error).lower()
        return any(key in text for key in (
            "bad gateway", "gateway", "timeout", "timed out",
            "connection", "http error 5", "service unavailable",
        ))

    def _init_vosk(self):
        try:
            import vosk
            import pyaudio
            vosk.SetLogLevel(-1)
            self._vosk_recognizer = vosk.KaldiRecognizer(
                vosk.Model(self._model_path), config.STT_SAMPLE_RATE
            )
            self._vosk_recognizer.SetWords(True)
            self._vosk_audio = pyaudio.PyAudio()
            self._vosk_stream = self._vosk_audio.open(
                format=pyaudio.paInt16, channels=1,
                rate=config.STT_SAMPLE_RATE, input=True,
                frames_per_buffer=config.STT_FRAMES_PER_BUFFER,
            )
            self._backend = "vosk"
            print("[STT] 使用 Vosk 离线识别 (中文) — 始终监听模式")
            return True
        except ImportError:
            print("[STT] Vosk 未安装")
        except Exception as e:
            print(f"[STT] Vosk 初始化失败: {e}")
        return False

    def _can_reply(self):
        return time.time() - self._last_reply_time >= config.STT_REPLY_COOLDOWN

    def _normalize_text(self, text):
        text = re.sub(r"\s+", "", text or "")
        text = text.replace("，", "").replace("。", "")
        text = re.sub(r"([\u4e00-\u9fff])\1{2,}", r"\1", text)
        return text.strip(" ，。！？,.!?;；：:")

    def _is_noise_text(self, text):
        if text in WAKE_ONLY_PATTERNS:
            return True
        if text in NOISE_FRAGMENTS:
            return True
        if len(text) <= 3 and any(word in text for word in ("歌", "曲", "副驾")):
            return True
        if len(text) <= 3 and all(char in FILLER_CHARS for char in text):
            return True
        if len(text) <= 6 and text.endswith("的") and not any(
            word in text for word in ("什么", "怎么", "哪里", "为何", "为什么", "开车", "疲劳", "状态")
        ):
            return True
        return False

    def _has_intent_hint(self, text):
        return any(hint in text for hint in INTENT_HINTS)

    def _is_assistant_echo(self, text):
        if len(text) < ASSISTANT_ECHO_MIN_LEN:
            return False
        if not self._assistant_text:
            return False
        if time.time() - self._assistant_text_time > config.STT_ASSISTANT_ECHO_WINDOW:
            return False

        assistant_text = self._assistant_text
        if text in assistant_text:
            return True
        if len(text) <= 12:
            ratio = SequenceMatcher(None, text, assistant_text).ratio()
            shared = sum(1 for char in set(text) if char in assistant_text)
            coverage = shared / max(len(set(text)), 1)
            return ratio >= 0.42 or coverage >= 0.75
        return False

    def _handle_speech(self, text):
        """识别到语音时调用。过滤过短文本，检查冷却期。"""
        text = self._normalize_text(text)
        if len(text) < 2:
            if text:
                print(f"[STT] 忽略过短: '{text}' ({len(text)}字)")
            return

        if (
            self._audio_level
            and self._audio_level < config.STT_MIN_AUDIO_LEVEL
            and not self._has_intent_hint(text)
        ):
            print(f"[STT] 忽略低音量: '{text}' level={self._audio_level:.4f}")
            return

        if self._is_noise_text(text):
            print(f"[STT] 忽略唤醒/噪声片段: '{text}'")
            return

        if self._is_assistant_echo(text):
            print(f"[STT] 忽略副驾播报回声: '{text}'")
            return

        now = time.time()
        if text == self._last_text and now - self._last_text_time < 4.0:
            print(f"[STT] 忽略重复: '{text}'")
            return

        if not self._can_reply():
            remaining = config.STT_REPLY_COOLDOWN - (time.time() - self._last_reply_time)
            print(f"[STT] 冷却中，忽略: '{text}' (剩余{remaining:.1f}s)")
            return

        print(f"[STT] 识别: {text}")
        self._last_reply_time = now
        self._last_text = text
        self._last_text_time = now
        if self._on_user_speech:
            self._on_user_speech(text)

    def _loop(self):
        print("[STT] 开始监听...")
        if self._backend == "google":
            self._loop_google()
        elif self._backend == "vosk":
            self._loop_vosk()

    # === Google 后端循环 ===
    def _loop_google(self):
        import speech_recognition as sr
        _google_fail_count = 0
        while self._running:
            if self._muted or time.time() < self._mute_until:
                time.sleep(0.3)
                continue

            try:
                with self._microphone as source:
                    audio = self._recognizer.listen(
                        source, timeout=1.0, phrase_time_limit=5.0,
                    )
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[STT] 监听错误: {e}")
                time.sleep(0.5)
                continue

            # 音频电平
            try:
                raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                self._audio_level = float(np.abs(raw).mean() / 32768.0)
            except Exception:
                pass

            # 识别
            try:
                text = self._recognizer.recognize_google(audio, language="zh-CN")
                _google_fail_count = 0  # 成功，重置计数
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                _google_fail_count += 1
                if self._is_network_error(e):
                    if self._switch_to_vosk(f"Google 网络/API 错误: {e}"):
                        return
                print(f"[STT] Google API 错误 ({_google_fail_count}/3): {e}")
                if _google_fail_count >= 3 and self._switch_to_vosk("Google 连续失败3次"):
                    return
                time.sleep(2.0)
                continue
            except Exception as e:
                print(f"[STT] 识别失败: {e}")
                continue

            if text:
                self._handle_speech(text)

    # === Vosk 后端循环 ===
    def _loop_vosk(self):
        import json
        while self._running:
            try:
                data = self._vosk_stream.read(
                    config.STT_FRAMES_PER_BUFFER, exception_on_overflow=False
                )
            except Exception as e:
                print(f"[STT] 音频读取错误: {e}")
                continue

            if self._muted or time.time() < self._mute_until:
                continue

            if len(data) > 0:
                chunk = np.frombuffer(data, dtype=np.int16)
                self._audio_level = float(np.abs(chunk).mean() / 32768.0)

            if self._vosk_recognizer.AcceptWaveform(data):
                result = json.loads(self._vosk_recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    self._handle_speech(text)
