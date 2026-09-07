"""STT 诊断工具 — 快速测试麦克风和唤醒词."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import vosk, pyaudio, config

def log(msg):
    print(msg, flush=True)

log("=" * 50)
log("STT 快速诊断")
log(f"模型路径: {config.VOSK_MODEL_PATH}")
log(f"模型存在: {os.path.exists(config.VOSK_MODEL_PATH)}")

# 设备
log("\n音频输入设备:")
p = pyaudio.PyAudio()
default_in = p.get_default_input_device_info()
log(f"  默认输入: [{default_in['index']}] {default_in['name']}")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        mark = " ← 默认" if i == default_in['index'] else ""
        log(f"  [{i}] {info['name']}{mark}")

# 模型
log("\n加载模型...")
vosk.SetLogLevel(-1)
model = vosk.Model(config.VOSK_MODEL_PATH)
rec = vosk.KaldiRecognizer(model, 16000)
rec.SetWords(True)
log("  模型加载成功")

# 麦克风
log("\n打开麦克风...")
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                frames_per_buffer=1600)
log("  麦克风打开成功")

# 监听
log("\n检测语音 (5秒，请说话)...")
start = time.time()
while time.time() - start < 5:
    data = stream.read(1600, exception_on_overflow=False)
    if rec.AcceptWaveform(data):
        text = json.loads(rec.Result()).get("text", "").strip()
        if text:
            log(f"  识别: [{text}]")
            if "副驾" in text:
                log("  >>> 唤醒词检测到! <<<")
    else:
        partial = json.loads(rec.PartialResult()).get("partial", "").strip()
        if partial:
            log(f"  部分: {partial}")
            if "副驾" in partial:
                log("  >>> 唤醒词检测到! <<<")

stream.close()
p.terminate()
log("\n诊断完成")
