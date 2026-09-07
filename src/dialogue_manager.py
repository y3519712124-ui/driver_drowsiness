"""对话状态机 — 决定副驾什么时候说什么话。"""
import time
from enum import Enum

import config


class CoPilotState(Enum):
    IDLE = 0
    OBSERVING = 1
    ENGAGING = 2
    WARNING = 3
    CRITICAL = 4


class DialogueManager:
    def __init__(self):
        self.current_state = CoPilotState.IDLE
        self._last_action_time = {}
        self._last_face_seen = time.time()
        self._state_entry_time = time.time()
        self._conv_history = []       # [(role, text), ...]

    def update(self, fatigue_analysis, voice_engine, llm_engine, conv_lib, alert_manager):
        """根据疲劳分析结果决定动作。返回执行的动作描述字符串。"""
        now = time.time()

        if fatigue_analysis is None:
            return None

        # 人脸长时间未检测到，不主动说话，避免摄像头没人时还在打扰。
        if now - self._last_face_seen > config.FACE_LOSS_TIMEOUT:
            return None

        # 根据分数确定目标状态
        target_state = self._score_to_state(fatigue_analysis.level)

        # 迟滞：降级时需要维持一段稳定时间
        if target_state.value < self.current_state.value:
            if now - self._state_entry_time < 5.0:
                target_state = self.current_state

        # 状态切换
        if target_state != self.current_state:
            self.current_state = target_state
            self._state_entry_time = now

        # 检查冷却
        if not self._can_act(now):
            return None

        actions = {
            CoPilotState.IDLE: lambda: None,
            CoPilotState.OBSERVING: lambda: self._act_observing(voice_engine, llm_engine, conv_lib, fatigue_analysis),
            CoPilotState.ENGAGING: lambda: self._act_engaging(voice_engine, llm_engine, conv_lib, fatigue_analysis),
            CoPilotState.WARNING: lambda: self._act_warning(voice_engine, llm_engine, conv_lib, fatigue_analysis),
            CoPilotState.CRITICAL: lambda: self._act_critical(voice_engine, alert_manager, fatigue_analysis),
        }

        result = actions[self.current_state]()
        if result:
            self._last_action_time[fatigue_analysis.level] = now
        return result

    def mark_face_seen(self):
        self._last_face_seen = time.time()

    def mark_face_lost(self):
        pass

    # === 内部方法 ===

    def _score_to_state(self, level):
        mapping = {0: CoPilotState.IDLE, 1: CoPilotState.OBSERVING,
                   2: CoPilotState.ENGAGING, 3: CoPilotState.WARNING,
                   4: CoPilotState.CRITICAL}
        return mapping.get(level, CoPilotState.IDLE)

    def _can_act(self, now):
        """检查当前状态是否已过冷却期。"""
        level_cfg = config.FATIGUE_LEVELS[self.current_state.value]
        cooldown = level_cfg["cooldown"]
        last = self._last_action_time.get(self.current_state.value, 0)
        return (now - last) >= cooldown

    def _get_text(self, llm_engine, conv_lib, prompt_type, category, factors):
        """优先 LLM，回退模板。"""
        text = None
        if config.LLM_PROACTIVE_ENABLED and llm_engine:
            text = llm_engine.generate(prompt_type, {"factors": factors})
        if text:
            return text
        return conv_lib.pick(category, factors)

    def _speak(self, voice_engine, text):
        if voice_engine and voice_engine.running:
            voice_engine.speak(text)

    def _act_observing(self, voice, llm, lib, analysis):
        text = self._get_text(llm, lib, "observation", "observations", analysis.factors)
        self._speak(voice, text)
        return f"[观察] {text}"

    def _act_engaging(self, voice, llm, lib, analysis):
        text = self._get_text(llm, lib, "engagement", "engagements", analysis.factors)
        self._speak(voice, text)
        return f"[互动] {text}"

    def _act_warning(self, voice, llm, lib, analysis):
        text = self._get_text(llm, lib, "warning", "warnings", analysis.factors)
        self._speak(voice, text)
        return f"[警告] {text}"

    def _act_critical(self, voice, alert, analysis):
        alert.trigger()
        text = self._conv_lib.pick("alerts") if self._conv_lib else "危险！请立刻靠边停车！"
        self._speak(voice, text)
        return f"[紧急] {text}"

    def _answer_local_question(self, user_text, fatigue_analysis):
        """常见驾驶安全问题的本地回答，避免 LLM 不可用时答非所问。"""
        text = user_text.replace("？", "").replace("?", "")

        if any(word in text for word in ("你是谁", "你叫什么", "你是干嘛", "你能做什么")):
            return "我是你的AI语音副驾，可以帮你监测疲劳、提醒休息、回答驾驶安全问题，也能控制音乐。"

        if any(word in text for word in ("夜车", "晚上开车", "开夜车")) and any(word in text for word in ("困", "疲劳", "容易")):
            return "夜车容易困，是因为生物钟进入休息时段，路况又单调，视觉刺激少，注意力会自然下降。"

        if any(word in text for word in ("什么时候", "啥时候", "何时")) and "疲劳" in text:
            return (
                "最容易疲劳的是凌晨两点到六点、午后一到三点、连续开车两小时后，"
                "还有饭后、夜间和高速长直路段。"
            )

        if any(word in text for word in ("为什么", "原因")) and "疲劳" in text:
            return "疲劳主要来自睡眠不足、连续驾驶太久、车内闷热、路线单调和饭后困倦。"

        if any(word in text for word in ("怎么办", "怎么缓解", "怎么提神", "困了", "犯困")):
            return "最有效是找安全位置停车休息十五分钟。开窗、喝水、活动肩颈只能临时提神。"

        if any(word in text for word in ("咖啡", "红牛", "功能饮料")):
            return "咖啡和功能饮料只能短时间提神，不能替代睡眠。已经明显犯困时，停车休息更安全。"

        if any(word in text for word in ("开窗", "吹风", "冷风")):
            return "开窗吹风能临时提神，但效果有限。如果眼皮发沉，还是要尽快找安全位置休息。"

        if any(word in text for word in ("多久休息", "多长时间休息", "开多久")):
            return "建议连续驾驶不超过两小时就休息一次，每次至少十到十五分钟。"

        if any(word in text for word in ("服务区", "停车", "靠边")) and any(word in text for word in ("休息", "找", "哪里", "能不能")):
            return "我现在不能导航找服务区，但如果你已经犯困，请优先选择最近的安全停车点休息。"

        if any(word in text for word in ("危险吗", "危害", "严重吗")) and "疲劳" in text:
            return "危险。疲劳会让反应变慢、判断变差，严重时会微睡眠，几秒钟就可能偏离车道。"

        if any(word in text for word in ("这首歌", "什么歌", "歌叫什么", "歌曲名")):
            return "车上这首歌叫《她是我的》。你可以说播放、暂停或关闭这首歌。"

        if any(word in text for word in ("讲个笑话", "笑话", "脑筋急转弯")):
            return "脑筋急转弯：什么东西越洗越脏？答案是水。好了，笑一下，眼睛也别离开路面。"

        if any(word in text for word in ("天气", "下雨", "下雪", "气温")):
            return "我现在不能联网查实时天气。开车时如果遇到雨雪雾，请降低车速、拉大车距。"

        if any(word in text for word in ("导航", "路线", "规划路线", "怎么走", "到哪里")):
            return "我现在不能做实时导航。路线请以车机或地图为准，我负责帮你盯疲劳和安全状态。"

        if any(word in text for word in ("我现在", "当前", "状态")) and any(word in text for word in ("怎么样", "困", "疲劳")):
            if not fatigue_analysis:
                return "我还在收集你的状态数据，请让脸部保持在画面里。"
            label = config.FATIGUE_LEVELS[fatigue_analysis.level]["label"]
            factors = "、".join(fatigue_analysis.factors[:2])
            return f"你现在是{label}状态，评分{fatigue_analysis.score:.0f}，主要信号是{factors}。"

        if any(word in text for word in ("谢谢", "谢了", "好的", "知道了", "明白")):
            return "不客气，我继续帮你看着状态。安全第一。"

        return None

    def respond_to_user(self, user_text, voice_engine, llm_engine, conv_lib, fatigue_analysis):
        """响应用户语音输入。返回回复文本。"""
        factors = fatigue_analysis.factors if fatigue_analysis else []
        level = fatigue_analysis.level if fatigue_analysis else 0

        # 构建对话历史摘要
        history_text = ""
        if self._conv_history:
            recent = self._conv_history[-6:]  # 最近3轮
            history_text = "\n".join(f"{'司机' if r[0]=='user' else '副驾'}: {r[1]}" for r in recent)

        context = {
            "user_text": user_text,
            "fatigue_level": level,
            "fatigue_score": int(fatigue_analysis.score) if fatigue_analysis else 0,
            "factors": factors,
            "history": history_text,
        }

        reply = self._answer_local_question(user_text, fatigue_analysis)
        if not reply:
            reply = llm_engine.generate("dialogue", context) if llm_engine else None
        if not reply:
            import random
            replies = [
                "这个我先记下。开车时我们优先保证安全和清醒。",
                "我听到了。你可以问我状态、休息建议，或者让我播放音乐。",
                "明白。需要我判断疲劳状态时，直接问“我现在困了吗”。",
                "收到。路上有任何不舒服，优先找安全位置停下。",
            ]
            reply = random.choice(replies)

        # 记录对话历史
        self._conv_history.append(("user", user_text))
        self._conv_history.append(("assistant", reply))
        if len(self._conv_history) > 12:
            self._conv_history = self._conv_history[-12:]

        self._speak(voice_engine, reply)
        return reply

    def set_conv_lib(self, lib):
        self._conv_lib = lib
