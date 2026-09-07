"""AI 语音副驾 — 驾驶疲劳检测系统。"""
import sys
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

WINDOW_TITLE = "AI Co-Pilot - Drowsiness Detection"

# 修复 Windows 控制台中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.system("chcp 65001 >nul" if os.name == "nt" else "")

import argparse
import time
import cv2
import statistics

import config
from src.camera import CameraManager
from src.face_detector import FaceDetector
from src.metrics_computer import compute_both_ear, compute_mar, estimate_head_pose, get_blendshape_scores
from src.metrics_buffer import MetricsBuffer, FatigueFrameMetrics
from src.overlay_renderer import OverlayRenderer
from src.voice_engine import VoiceEngine
from src.llm_engine import LLMEngine
from src.conversation_library import ConversationLibrary
from src.alert_manager import AlertManager
from src.co_pilot_brain import CoPilotBrain
from src.startup_check import run_startup_checks, print_startup_checks
from src.session_logger import SessionLogger


def parse_args():
    p = argparse.ArgumentParser(description="AI 语音副驾 — 驾驶疲劳检测")
    p.add_argument("--no-voice", action="store_true", help="禁用语音")
    p.add_argument("--no-llm", action="store_true", help="禁用 LLM，仅用预制模板")
    p.add_argument("--debug", action="store_true", help="打印调试信息")
    p.add_argument("--calibrate", action="store_true", help="校准个人 EAR 基线")
    p.add_argument("--offline-stt", action="store_true", help="强制使用 Vosk 离线语音识别")
    p.add_argument("--online-stt", action="store_true", help="使用 Google 在线语音识别，失败后回退 Vosk")
    return p.parse_args()


def calibrate_ear_threshold(cam, detector, overlay, seconds=8.0):
    """采集个人睁眼 EAR 基线，并返回动态闭眼阈值。"""
    samples = []
    start = time.time()
    print(f"开始校准 EAR，请自然睁眼看向前方 {seconds:.0f} 秒...")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        elapsed = time.time() - start
        remaining = max(0.0, seconds - elapsed)
        result = detector.detect(frame)

        if result.face_detected:
            w, h = result.image_width, result.image_height
            lm = result.landmarks
            _, _, avg_ear = compute_both_ear(lm, w, h)
            if 0.12 < avg_ear < 0.45:
                samples.append(avg_ear)
            overlay.draw_face_landmarks(frame, lm, w, h)
            overlay.draw_eye_landmarks(frame, lm, w, h)
            overlay.draw_ear(frame, avg_ear, True)
            overlay.draw_calibration(frame, remaining, len(samples))
        else:
            overlay.draw_no_face(frame)
            overlay.draw_calibration(frame, remaining, len(samples), "请让脸部进入画面")

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if elapsed >= seconds:
            break

    if len(samples) < 20:
        print("校准样本不足，继续使用默认 EAR 阈值。")
        return None

    baseline = statistics.median(samples)
    threshold = max(0.15, min(0.28, baseline * 0.72))
    print(f"校准完成：EAR 基线 {baseline:.3f}，闭眼阈值 {threshold:.3f}")
    return threshold


def main():
    args = parse_args()
    print_startup_checks(run_startup_checks(
        voice_enabled=not args.no_voice,
        llm_enabled=not args.no_llm,
    ))

    # 初始化各模块
    detector = FaceDetector()
    overlay = OverlayRenderer()
    buffer = MetricsBuffer()

    voice = VoiceEngine()
    if not args.no_voice:
        voice.start()
        voice.speak("AI语音副驾已上线，请安全驾驶。")

    llm = LLMEngine()
    if not args.no_llm:
        llm.start()

    conv_lib = ConversationLibrary()
    alert_mgr = AlertManager()

    brain = CoPilotBrain(
        buffer, voice, llm, conv_lib, alert_mgr,
        voice_enabled=not args.no_voice,
        stt_backend="google" if args.online_stt and not args.offline_stt else "vosk",
    )
    brain.start()
    session = SessionLogger()

    try:
        with CameraManager(config.CAMERA_INDEX) as cam:
            ear_threshold = config.EAR_THRESHOLD
            if args.calibrate:
                calibrated = calibrate_ear_threshold(cam, detector, overlay)
                if calibrated is None:
                    print("校准未完成，使用默认阈值。")
                else:
                    ear_threshold = calibrated

            session.log_start(ear_threshold, args)

            print("AI 语音副驾已启动（MediaPipe FaceLandmarker）")
            print("按 q 退出")
            print(f"EAR 阈值: {ear_threshold:.3f}，闭眼超 {config.WAIT_TIME}s 报警")
            if args.no_voice:
                print("语音已禁用")
            if args.no_llm:
                print("LLM 已禁用，使用预制模板")
            print()

            d_time = 0.0
            closed_frames = 0
            t1 = time.time()

            while True:
                frame = cam.read_frame()
                if frame is None:
                    break

                result = detector.detect(frame)
                w, h = result.image_width, result.image_height

                # 构建帧指标
                metrics = FatigueFrameMetrics(
                    timestamp=time.time(),
                    face_detected=result.face_detected,
                )

                if result.face_detected:
                    lm = result.landmarks

                    # 绘制
                    overlay.draw_face_landmarks(frame, lm, w, h)
                    overlay.draw_eye_landmarks(frame, lm, w, h)

                    # 计算所有指标
                    left_ear, right_ear, avg_ear = compute_both_ear(lm, w, h)
                    mar = compute_mar(lm, w, h)
                    pitch, _ = estimate_head_pose(lm, w, h)
                    bs = get_blendshape_scores(result.blendshapes)

                    metrics.left_ear = left_ear
                    metrics.right_ear = right_ear
                    metrics.avg_ear = avg_ear
                    metrics.mar = mar
                    metrics.head_pitch = pitch
                    metrics.eye_blink_left = bs["eyeBlinkLeft"]
                    metrics.eye_blink_right = bs["eyeBlinkRight"]
                    metrics.jaw_open = bs["jawOpen"]

                    # 判断闭眼
                    if avg_ear < ear_threshold:
                        closed_frames += 1
                    else:
                        closed_frames = 0

                    eyes_detected = closed_frames < config.CONSEC_FRAMES
                    metrics.eyes_closed = not eyes_detected

                    # 显示
                    overlay.draw_status(frame, "AWAKE" if eyes_detected else "CLOSED")
                    overlay.draw_ear(frame, avg_ear, eyes_detected)
                    overlay.draw_blendshapes(frame, result.blendshapes)

                    # 闭眼计时 + 硬报警
                    t2 = time.time()
                    if not eyes_detected:
                        d_time += (t2 - t1)
                        overlay.draw_closed_timer(frame, d_time)
                    else:
                        d_time = 0
                    t1 = t2

                    if d_time >= config.WAIT_TIME:
                        overlay.draw_alert(frame, "DROWSINESS ALERT!")
                        if alert_mgr.trigger():
                            session.record_alert(d_time)
                    else:
                        alert_mgr.reset()

                    # 疲劳等级显示
                    overlay.draw_fatigue_level(frame, brain.level, brain.score)
                    overlay.draw_co_pilot_status(frame, brain.state)
                    overlay.draw_stt_hint(frame)
                    overlay.draw_conversation(frame, brain.last_user_text, brain.last_reply_text)

                    # 麦克风指示灯（右上角，含音频电平）
                    audio_level = brain.audio_level
                    overlay.draw_mic_indicator(
                        frame, not args.no_voice, audio_level, backend=brain.stt_backend
                    )

                    if args.debug:
                        print(f"EAR={avg_ear:.3f} MAR={mar:.3f} Pitch={pitch:.1f} "
                              f"BlinkL={bs['eyeBlinkLeft']:.2f} BlinkR={bs['eyeBlinkRight']:.2f} "
                              f"Fatigue Lv{brain.level}({brain.score:.0f})")

                else:
                    overlay.draw_no_face(frame)
                    d_time = 0
                    closed_frames = 0

                # 推入缓冲区供副驾分析
                buffer.push(metrics)
                session.update_frame(metrics, brain.level, brain.score)
                session.update_voice(brain.last_user_text, brain.last_reply_text)
                session.update_stt_backend(brain.stt_backend)

                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("已退出")
                    break
                elif key == 13:
                    pass  # Enter — 不再需要手动唤醒（始终监听）
    finally:
        summary = session.finish(buffer.get_snapshot())
        print()
        print(summary)
        print(f"日志文件: {session.log_path}")
        print(f"摘要文件: {session.summary_path}")
        brain.stop()
        voice.stop()
        llm.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
