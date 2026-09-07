"""LLM 引擎 — 支持 Ollama（优先）和 llama-cpp-python 两种后端。"""
import os
import threading
import queue
import json
import time
import urllib.request
import urllib.error

import config

# Ollama API 端点
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"

# 副驾知识库 — 持久偏好
COPILOT_KNOWLEDGE = (
    "你的知识库：车上只有一首歌叫《她是我的》。"
    "当驾驶员聊到音乐或听歌时，只聊这首歌，不要提到其他歌曲名字。"
    "注意：系统会自动处理音乐的播放、暂停、停止，你不需要控制音乐。"
    "你只需要自然地聊天，像副驾朋友一样回应驾驶员说的话。"
)


class LLMEngine:
    """LLM 推理引擎，运行在后台线程中。

    优先检测 Ollama（Windows 上最简单），其次尝试 llama-cpp-python。
    调用方通过 self.available 判断是否可用。
    """

    def __init__(self, backend="auto"):
        self.available = False
        self._backend = None       # "ollama" | "llama_cpp"
        self._model = None         # llama-cpp-python model object
        self._ollama_model = None  # 实际的 Ollama 模型名
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = None
        self._running = False
        self._preferred = backend
        self._request_id = 0
        self._failure_count = 0
        self._cooldown_until = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="LLMEngine")
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._request_queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def generate(self, prompt_type: str, context: dict | None = None) -> str | None:
        """异步请求 LLM 生成。返回 None 表示未就绪。"""
        if not self.available or time.time() < self._cooldown_until:
            return None
        context = context or {}
        self._request_id += 1
        request_id = self._request_id
        self._drain_stale_responses()
        self._request_queue.put((request_id, prompt_type, context))

        deadline = time.time() + config.LLM_TIMEOUT
        while time.time() < deadline:
            timeout = max(0.1, deadline - time.time())
            try:
                response_id, result = self._response_queue.get(timeout=timeout)
            except queue.Empty:
                break
            self._response_queue.task_done()
            if response_id == request_id:
                return result
            print(f"[LLMEngine] 丢弃过期响应: #{response_id}")

        self._mark_failure("请求超时", noisy=False)
        return None

    # === 内部 ===

    def _loop(self):
        self._try_connect()
        while self._running:
            try:
                item = self._request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            request_id, prompt_type, context = item
            try:
                if self._backend == "ollama":
                    result = self._generate_ollama(prompt_type, context)
                else:
                    result = self._generate_llama_cpp(prompt_type, context)
                self._failure_count = 0
                self._response_queue.put((request_id, result))
            except Exception as e:
                reason = str(e) or e.__class__.__name__
                is_timeout = "timed out" in reason.lower() or isinstance(e, TimeoutError)
                if not is_timeout:
                    print(f"[LLMEngine] 生成失败: {reason}")
                self._mark_failure(reason, noisy=not is_timeout)
                self._response_queue.put((request_id, None))

            self._request_queue.task_done()

    def _drain_stale_responses(self):
        try:
            while True:
                self._response_queue.get_nowait()
                self._response_queue.task_done()
        except queue.Empty:
            pass

    def _mark_failure(self, reason, noisy=True):
        self._failure_count += 1
        if self._failure_count >= 3:
            self._cooldown_until = time.time() + 15.0
            if noisy:
                print(f"[LLMEngine] 连续失败，暂停 LLM 15 秒: {reason}")
            else:
                print("[LLMEngine] LLM 响应较慢，短暂暂停并使用本地回复。")

    def _try_connect(self):
        # 1) 尝试 Ollama
        if self._preferred in ("auto", "ollama"):
            if self._check_ollama():
                self._backend = "ollama"
                self.available = True
                print(f"[LLMEngine] Ollama 已连接，模型: {self._ollama_model}")
                threading.Thread(target=self._warmup_ollama, daemon=True, name="LLMWarmup").start()
                return

        # 2) 尝试 llama-cpp-python
        if self._preferred in ("auto", "llama_cpp"):
            if self._load_llama_cpp():
                self._backend = "llama_cpp"
                self.available = True
                print("[LLMEngine] llama-cpp-python 模型加载成功")
                return

        print("[LLMEngine] LLM 不可用，将使用预制对话模板。")
        print("[LLMEngine] 推荐安装 Ollama：https://ollama.com/download/windows")
        print("[LLMEngine] 然后运行：ollama pull qwen2.5:1.5b")

    def _check_ollama(self):
        """检测 Ollama 是否运行，自动选择最佳可用模型（优先小模型）。"""
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if not models:
                    print("[LLMEngine] Ollama 运行中但没有模型。运行: ollama pull qwen2.5:1.5b")
                    return False

                # 优先匹配：qwen2.5:1.5b > qwen2.5:3b > qwen2.5:7b > 任意 qwen
                prefer = ["qwen2.5:1.5b", "qwen2.5:0.5b", "qwen2.5:3b", "qwen2.5:7b"]
                for p in prefer:
                    if p in models:
                        self._ollama_model = p
                        return True

                # 回退：任意 qwen 系列
                for m in models:
                    if "qwen" in m.lower() and "embed" not in m.lower():
                        self._ollama_model = m
                        print(f"[LLMEngine] 使用可用模型: {m}")
                        return True

                print(f"[LLMEngine] Ollama 模型列表: {models}")
                print("[LLMEngine] 未找到 qwen 系列模型。运行: ollama pull qwen2.5:1.5b")
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            print("[LLMEngine] Ollama 未运行。安装: https://ollama.com/download/windows")
        except Exception as e:
            print(f"[LLMEngine] Ollama 检测异常: {e}")
        return False

    def _warmup_ollama(self):
        try:
            payload = json.dumps({
                "model": self._ollama_model or OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": "你好"},
                ],
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 1, "num_ctx": 512},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60):
                pass
            print("[LLMEngine] Ollama 模型预热完成")
        except Exception as e:
            print(f"[LLMEngine] Ollama 预热跳过: {e}")

    def _load_llama_cpp(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            return False

        model_path = os.path.join(config.MODEL_DIR, "models", config.LLM_MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"[LLMEngine] 模型文件不存在: {model_path}")
            return False

        try:
            self._model = Llama(model_path=model_path, n_ctx=1024, n_threads=4, verbose=False)
            return True
        except Exception as e:
            print(f"[LLMEngine] llama-cpp 加载失败: {e}")
            return False

    def _generate_ollama(self, prompt_type, context):
        system, user_msg = self._build_prompt(prompt_type, context)
        payload = json.dumps({
            "model": self._ollama_model or OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg or "请说点什么。"},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.6,
                "num_predict": 32,
                "num_ctx": 1024,
                "top_p": 0.9,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            data = json.loads(resp.read())
            text = data["message"]["content"].strip()
            if text and not text.endswith(("。", "！", "？", "!", "?", "~")):
                text += "。"
            return text

    def _generate_llama_cpp(self, prompt_type, context):
        if not self._model:
            return None
        system, user_msg = self._build_prompt(prompt_type, context)
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg or "请说点什么。"},
            ],
            max_tokens=80,
            temperature=0.7,
        )
        text = response["choices"][0]["message"]["content"].strip()
        if text and not text.endswith(("。", "！", "？", "!", "?", "~")):
            text += "。"
        return text

    def _build_prompt(self, prompt_type, context):
        prompts = {
            "observation": (
                "你是一个友好的AI副驾助手，正在和驾驶员聊天帮助他保持清醒。"
                "驾驶员当前处于轻度疲劳状态。请用中文说一句简短的关心或观察（15-30个字），"
                "语气自然像朋友聊天，不要啰嗦。\n"
                + COPILOT_KNOWLEDGE
            ),
            "engagement": (
                "你是一个幽默的AI副驾助手，正在和驾驶员互动帮他提神。"
                "驾驶员当前处于中度疲劳状态。请用中文出一个有趣的脑筋急转弯、"
                "讲一个笑话、或者问一个有意思的问题（20-40字）。要轻松好玩。\n"
                + COPILOT_KNOWLEDGE
            ),
            "warning": (
                "你是一个负责任的AI副驾助手。驾驶员当前处于重度疲劳状态，有安全风险。"
                "请用中文说一句严肃但关切的提醒，建议驾驶员立即休息（15-30字）。"
                "语气坚定但不吓人。\n"
                + COPILOT_KNOWLEDGE
            ),
            "dialogue": (
                "你是一个在汽车里的AI副驾助手，正在和驾驶员对话。你关心驾驶员的状态。"
                "请用中文简短自然地回复驾驶员（15-40字），像一个坐在副驾驶的朋友。"
                "语气轻松友好。\n"
                + COPILOT_KNOWLEDGE
            ),
        }
        system = prompts.get(prompt_type, prompts["observation"])

        # dialogue 类型：用户说的话在 context 中
        if prompt_type == "dialogue":
            user_text = context.get("user_text", "")
            history = context.get("history", "")
            level = context.get("fatigue_level", 0)
            level_labels = {0: "清醒", 1: "轻度疲劳", 2: "中度疲劳", 3: "重度疲劳", 4: "危险"}
            level_label = level_labels.get(level, "未知")
            user_msg = f"驾驶员当前状态：{level_label}。"
            if history:
                user_msg += f"\n对话历史：\n{history}\n"
            user_msg += f"驾驶员说：{user_text}"
        else:
            user_msg = ""
            if context.get("factors"):
                user_msg = f"检测到的疲劳信号：{'、'.join(context['factors'])}。"
        return system, user_msg
