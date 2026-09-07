"""启动自检：检查演示所需的关键资源。"""
import os
import socket
import json
import urllib.request

import cv2

import config


def _check_camera(index):
    cap = cv2.VideoCapture(index)
    ok = cap.isOpened()
    if ok:
        ok, _ = cap.read()
    cap.release()
    return bool(ok), f"摄像头 {index}" if ok else f"摄像头 {index} 不可用或被占用"


def _check_file(path, label):
    ok = os.path.exists(path)
    return ok, f"{label}: {path}" if ok else f"{label} 缺失: {path}"


def _check_microphone():
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        count = pa.get_device_count()
        has_input = any(
            pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
            for i in range(count)
        )
        pa.terminate()
        return has_input, "麦克风可用" if has_input else "未发现可用麦克风"
    except Exception as e:
        return False, f"麦克风检查失败: {e}"


def _check_ollama():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            chat_models = [m for m in models if "embed" not in m.lower()]
            if chat_models:
                return True, f"Ollama 已运行，模型: {', '.join(chat_models[:3])}"
            return False, "Ollama 已运行，但没有聊天模型；运行: ollama pull qwen2.5:1.5b"
    except Exception:
        return False, "Ollama 未运行，LLM 将回退模板"


def _check_port(host="localhost", port=11434):
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def run_startup_checks(voice_enabled=True, llm_enabled=True):
    """返回自检结果列表：[(ok, message), ...]。"""
    root = config.MODEL_DIR
    music_file = os.path.join(root, "M800001U9RTs08qoj9.mp3")
    checks = [
        _check_camera(config.CAMERA_INDEX),
        _check_file(config.MODEL_PATH, "FaceLandmarker 模型"),
        _check_file(config.VOSK_MODEL_PATH, "Vosk 中文模型"),
        _check_file(music_file, "音乐文件"),
    ]

    if voice_enabled:
        checks.append(_check_microphone())
    if llm_enabled:
        checks.append(_check_ollama() if _check_port() else (False, "Ollama 未运行，LLM 将回退模板"))

    return checks


def print_startup_checks(checks):
    print("启动自检：")
    for ok, message in checks:
        mark = "OK" if ok else "--"
        print(f"  [{mark}] {message}")
    print()
