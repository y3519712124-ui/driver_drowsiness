"""麦克风底层诊断 — 测试音频输入是否正常。"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time

print("=" * 50)
print("麦克风诊断")
print("=" * 50)

# 1. speech_recognition 设备列表
print("\n[1] speech_recognition 麦克风列表:")
try:
    import speech_recognition as sr
    names = sr.Microphone.list_microphone_names()
    for i, name in enumerate(names):
        print(f"    [{i}] {name}")
    default_mic = sr.Microphone()
    print(f"    默认: [{default_mic.device_index}]")
except Exception as e:
    print(f"    错误: {e}")

# 2. pyaudio 设备列表
print("\n[2] PyAudio 设备列表:")
try:
    import pyaudio
    p = pyaudio.PyAudio()
    default_in = p.get_default_input_device_info()
    print(f"    默认输入设备: [{default_in['index']}] {default_in['name']}")
    print(f"    默认采样率: {default_in['defaultSampleRate']}")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"    [{i}] {info['name']} (输入通道:{info['maxInputChannels']} 采样率:{info['defaultSampleRate']})")
    p.terminate()
except Exception as e:
    print(f"    错误: {e}")

# 3. 尝试录音5秒保存文件
print("\n[3] 录音测试 (5秒)... 请对着麦克风说话...")
try:
    import pyaudio
    import wave
    import numpy as np

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                    input=True, frames_per_buffer=1600)
    stream.start_stream()

    frames = []
    max_level = 0
    for _ in range(50):  # 5 seconds
        data = stream.read(1600, exception_on_overflow=False)
        frames.append(data)
        level = np.abs(np.frombuffer(data, dtype=np.int16)).mean() / 32768.0
        max_level = max(max_level, level)
        bar = "#" * int(level * 50)
        print(f"    电平: {level:.4f} {bar}", end="\r")

    stream.stop_stream()
    stream.close()
    p.terminate()

    print(f"\n    最大电平: {max_level:.4f}")

    # 保存录音文件供检查
    output_path = "mic_test.wav"
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"    录音已保存到: {output_path}")

    if max_level < 0.01:
        print("\n⚠ 警告：最大电平极低 ({:.4f})，麦克风可能没有拾取到声音！".format(max_level))
        print("  可能原因：")
        print("  1. Windows 麦克风隐私设置阻止了 Python 访问")
        print("  2. 默认麦克风不是你在用的设备")
        print("  3. 麦克风硬件静音或音量太低")
        print("\n  解决方法：")
        print("  - Win+I → 隐私 → 麦克风 → 确保'允许应用访问麦克风'已开启")
        print("  - 声音设置 → 输入 → 选择正确的麦克风设备")
        print("  - 说话时观察 Windows 声音设置中的输入电平条是否跳动")
    else:
        print(f"\n✓ 麦克风工作正常（最大电平: {max_level:.4f}）")
        print("  如果仍然无法识别，可能是语音识别准确度问题，不是麦克风问题。")

except Exception as e:
    print(f"\n    录音失败: {e}")
    import traceback
    traceback.print_exc()
