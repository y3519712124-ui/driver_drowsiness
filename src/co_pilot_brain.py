"""副驾大脑 — 后台线程，串联所有子系统。"""
import os
import queue
import threading
import time

import pygame.mixer

import config
from src.fatigue_analyzer import FatigueAnalyzer
from src.dialogue_manager import DialogueManager
from src.stt_engine import STTEngine

# 驾驶员专属歌曲
MUSIC_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "M800001U9RTs08qoj9.mp3")

# 音乐指令 — 按优先级排列，长匹配在前防止误判
STOP_PATTERNS = [
    "不要播放", "别播放", "不要放了", "别放了", "不要放", "别放",
    "关了", "关掉", "关闭", "停了", "停掉", "别唱了", "不要唱",
    "关闭这首歌", "关掉这首歌", "停这首歌", "停止这首歌", "别放这首歌",
]
PAUSE_PATTERNS = ["暂停", "停一下", "停一停"]
RESUME_PATTERNS = ["继续放", "接着放", "继续播", "继续唱"]
PLAY_PATTERNS = [
    "放一首歌", "放首歌", "来一首歌", "来首歌", "给我放", "为我放",
    "打开音乐", "开音乐", "放歌", "听歌", "播放", "放音乐", "来首", "放一下",
    "放", "播", "唱首歌",
]
STATUS_PATTERNS = ["状态怎么样", "现在状态", "我困了吗", "疲劳程度", "精神怎么样"]
QUIET_PATTERNS = ["安静一点", "先别说话", "别提醒", "少说话", "静音副驾"]
RESUME_TALK_PATTERNS = ["继续提醒", "恢复提醒", "可以说话", "打开副驾", "继续说话"]
WAKE_WORD_VARIANTS = [
    "副驾", "副驾驶", "副家", "付驾", "附加", "副将", "复制啊",
    "嫁为", "驾为", "小副驾",
]
MUSIC_TOPIC_FRAGMENTS = ["这首歌", "首歌", "歌曲", "音乐"]


class CoPilotBrain:
    def __init__(
        self, metrics_buffer, voice_engine, llm_engine, conv_library,
        alert_manager, voice_enabled=True, stt_backend="auto",
    ):
        self._buffer = metrics_buffer
        self._voice = voice_engine
        self._llm = llm_engine
        self._lib = conv_library
        self._alert = alert_manager
        self._voice_enabled = voice_enabled
        self._analyzer = FatigueAnalyzer()
        self._dialogue = DialogueManager()
        self._dialogue.set_conv_lib(conv_library)

        self._thread = None
        self._running = False
        self._current_score = 0
        self._current_level = 0
        self._last_analysis = None
        self._last_user_text = ""
        self._last_reply_text = ""
        self._music_loaded = False
        self._music_paused = False
        self._quiet_until = 0.0

        # 对话活跃窗口 — 用户说话后 N 秒内抑制主动搭话
        self._last_user_interaction = 0
        self._user_reply_pending = False  # LLM 正在生成回复中

        # 用户语音队列 — STT 线程放入，CoPilotBrain 线程处理
        self._speech_queue = queue.Queue(maxsize=3)

        self._stt = STTEngine(backend=stt_backend) if voice_enabled else None
        if self._stt:
            self._stt.on_user_speech(self._on_user_speech)

    # === 属性 ===
    @property
    def state(self):
        return self._dialogue.current_state.name

    @property
    def score(self):
        return self._current_score

    @property
    def level(self):
        return self._current_level

    @property
    def audio_level(self):
        return self._stt.audio_level if self._stt else 0.0

    @property
    def stt_backend(self):
        return self._stt.backend if self._stt else "off"

    @property
    def last_user_text(self):
        return self._last_user_text

    @property
    def last_reply_text(self):
        return self._last_reply_text

    # === 公开方法 ===
    def start(self):
        if self._running:
            return
        pygame.mixer.init()
        self._running = True
        if self._stt:
            self._stt.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CoPilotBrain")
        self._thread.start()
        mode = "始终监听模式" if self._stt else "视觉提醒模式"
        print(f"[CoPilot] 副驾大脑已启动 — {mode}")

    def stop(self):
        self._running = False
        self._stop_music()
        pygame.mixer.quit()
        if self._stt:
            self._stt.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # === 音乐意图识别 ===
    def _detect_music_intent(self, text):
        """识别音乐指令意图。返回 'stop' | 'pause' | 'resume' | 'play' | None。"""
        for pat in STOP_PATTERNS:
            if pat in text:
                return "stop"
        for pat in PAUSE_PATTERNS:
            if pat in text:
                return "pause"
        for pat in RESUME_PATTERNS:
            if pat in text:
                return "resume"
        for pat in PLAY_PATTERNS:
            if pat in text:
                return "play"
        return None

    def _normalize_user_text(self, text):
        text = text.strip()
        for word in WAKE_WORD_VARIANTS:
            text = text.replace(word, "副驾")
        while text.startswith("副驾副驾"):
            text = text[2:]
        return text

    def _is_music_fragment(self, text):
        if any(fragment in text for fragment in MUSIC_TOPIC_FRAGMENTS):
            command_words = STOP_PATTERNS + PAUSE_PATTERNS + RESUME_PATTERNS + PLAY_PATTERNS
            return not any(word in text for word in command_words)
        return False

    def _detect_control_intent(self, text):
        """识别副驾控制命令。"""
        for pat in STATUS_PATTERNS:
            if pat in text:
                return "status"
        for pat in QUIET_PATTERNS:
            if pat in text:
                return "quiet"
        for pat in RESUME_TALK_PATTERNS:
            if pat in text:
                return "resume_talk"
        return None

    def _speak_ack(self, text):
        self._last_reply_text = text
        if self._voice_enabled:
            self._voice.speak(text)

    def _describe_current_status(self):
        labels = {0: "清醒", 1: "轻度疲劳", 2: "中度疲劳", 3: "重度疲劳", 4: "危险"}
        label = labels.get(self._current_level, "未知")
        if not self._last_analysis:
            return "我还在收集状态数据，请保持面部在画面里。"
        factors = "、".join(self._last_analysis.factors[:2])
        return f"当前是{label}，评分{self._current_score:.0f}。主要信号是{factors}。"

    # === STT 回调（在 STT 线程中调用，必须立即返回不阻塞）===
    def _on_user_speech(self, user_text):
        user_text = self._normalize_user_text(user_text)
        print(f"[CoPilot] 用户: {user_text}")
        self._last_user_text = user_text
        self._last_user_interaction = time.time()

        # 音乐指令立即执行
        intent = self._detect_music_intent(user_text)
        if intent == "stop":
            self._stop_music()
            self._speak_ack("好的，音乐停了。")
            return
        elif intent == "pause":
            self._pause_music()
            self._speak_ack("已暂停。")
            return
        elif intent == "resume":
            self._resume_music()
            self._speak_ack("继续播放。")
            return
        elif intent == "play":
            self._play_music()
            self._speak_ack("好，放这首《她是我的》。")
            return

        control = self._detect_control_intent(user_text)
        if control == "status":
            self._speak_ack(self._describe_current_status())
            return
        if control == "quiet":
            self._quiet_until = time.time() + 300.0
            self._speak_ack("好，我先安静五分钟，危险时还会提醒你。")
            return
        if control == "resume_talk":
            self._quiet_until = 0.0
            self._speak_ack("好，我继续帮你盯着状态。")
            return

        if self._is_music_fragment(user_text):
            self._last_reply_text = "音乐相关片段已忽略。"
            print(f"[CoPilot] 忽略音乐残片: {user_text}")
            return

        # 放入队列，由主循环异步处理 LLM 回复
        try:
            self._speech_queue.put_nowait(user_text)
        except queue.Full:
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
                self._speech_queue.put_nowait(user_text)
            except queue.Empty:
                pass

    # === 音乐控制 ===
    def _play_music(self):
        if not os.path.exists(MUSIC_FILE):
            print(f"[CoPilot] 歌曲文件不存在: {MUSIC_FILE}")
            return
        try:
            if self._music_loaded:
                pygame.mixer.music.stop()
            pygame.mixer.music.load(MUSIC_FILE)
            pygame.mixer.music.play()
            self._music_loaded = True
            self._music_paused = False
            print("[CoPilot] 播放歌曲: 她是我的")
        except Exception as e:
            print(f"[CoPilot] 播放失败: {e}")

    def _stop_music(self):
        if self._music_loaded:
            pygame.mixer.music.stop()
            self._music_loaded = False
            self._music_paused = False
            print("[CoPilot] 停止播放")

    def _pause_music(self):
        if self._music_loaded and not self._music_paused:
            pygame.mixer.music.pause()
            self._music_paused = True
            print("[CoPilot] 暂停播放")

    def _resume_music(self):
        if self._music_loaded and self._music_paused:
            pygame.mixer.music.unpause()
            self._music_paused = False
            print("[CoPilot] 继续播放")

    # === 主循环 ===
    def _loop(self):
        while self._running:
            time.sleep(0.3)

            # TTS 播放时静音麦克风，防止识别自己的声音
            if self._stt:
                if self._voice.speaking and self._voice.current_text:
                    self._stt.set_assistant_text(self._voice.current_text, time.time())
                elif self._voice.last_spoken_text:
                    self._stt.set_assistant_text(
                        self._voice.last_spoken_text,
                        self._voice.last_spoken_ended_at,
                    )
                self._stt.set_muted(self._voice.speaking)

            # === 处理用户语音（异步，不阻塞 STT 线程）===
            try:
                user_text = self._speech_queue.get_nowait()
            except queue.Empty:
                user_text = None

            if user_text:
                self._user_reply_pending = True
                reply = self._dialogue.respond_to_user(
                    user_text, self._voice, self._llm, self._lib, self._last_analysis
                )
                self._user_reply_pending = False
                if reply:
                    self._last_reply_text = reply
                    print(f"[CoPilot] 回复: {reply}")
                # 不再根据 LLM 回复自动播放音乐 — 只响应用户明确指令

            # === 疲劳检测 ===
            snapshot = self._buffer.get_snapshot()
            short_frames = snapshot.get("short_frames", [])
            if not short_frames:
                continue

            last = short_frames[-1]
            if last.face_detected:
                self._dialogue.mark_face_seen()
            else:
                self._dialogue.mark_face_lost()

            analysis = self._analyzer.analyze(snapshot)
            self._last_analysis = analysis
            self._current_score = analysis.score
            self._current_level = analysis.level

            # 对话活跃期抑制主动搭话
            # 再次检查队列，防止 STT 线程在 fatigue 检查间隙放入新语音
            if not self._speech_queue.empty():
                continue
            now = time.time()
            in_conversation = (now - self._last_user_interaction) < 15.0
            just_spoke = (now - self._last_user_interaction) < 3.0
            # 正在播放 TTS 时不插话
            if self._voice.speaking:
                continue
            if now < self._quiet_until and analysis.level < 3:
                continue
            if in_conversation or self._user_reply_pending or just_spoke:
                continue

            # 决定主动搭话动作
            result = self._dialogue.update(
                analysis, self._voice, self._llm, self._lib, self._alert
            )

            if result:
                self._last_reply_text = result.split("] ", 1)[-1]
                print(f"[CoPilot] Lv{analysis.level}({analysis.score:.0f}) {result}")
