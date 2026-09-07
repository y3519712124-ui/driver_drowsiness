# driver_drowsiness

基于 OpenCV 和 MediaPipe 的驾驶疲劳检测与 AI 语音副驾实验项目。

## 功能

- 摄像头人脸关键点检测，计算 EAR、MAR 和头部姿态。
- 综合闭眼、眨眼、微睡眠、打哈欠、点头和 PERCLOS 等指标评估疲劳等级。
- 实时画面叠加、个人 EAR 校准、Windows 蜂鸣报警。
- 中文语音播报、Vosk 离线识别和可选 Google 在线识别。
- 可选 Ollama / llama-cpp-python 本地对话；模型不可用时回退预制模板。
- 本地会话日志与摘要。

## 安装与运行

当前面向 Windows（报警直接依赖 winsound）。需要 Python 3.10 或以上和摄像头；语音功能还需要麦克风和扬声器。依赖尚未锁定版本，具体 Python / 依赖组合需在目标电脑验证。

~~~powershell
git clone https://github.com/y3519712124-ui/driver_drowsiness.git
cd driver_drowsiness
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py --no-voice --no-llm
~~~

首次运行按 config.py 中的地址自动下载 face_landmarker.task，需要联网；也可预先下载并放在项目根目录。启动自检先于下载，因此首次启动可能先提示模型缺失。

启用离线语音前，从 Vosk 官方模型页面 https://alphacephei.com/vosk/models 下载 vosk-model-small-cn-0.22，解压到 models/vosk-model-small-cn-0.22/，该目录下应直接包含 am、conf、graph、ivector 等子目录。

~~~powershell
.\venv\Scripts\python.exe main.py --offline-stt --no-llm
~~~

如需本地大模型对话，安装并运行 Ollama，然后执行：

~~~powershell
ollama pull qwen2.5:1.5b
.\venv\Scripts\python.exe main.py --offline-stt
~~~

另一可选后端是单独安装 llama-cpp-python，并把 config.py 指定的 GGUF 文件放进 models/。模型和该可选依赖不随仓库提供。

也可双击“启动.bat”。窗口中按 q 退出。

## 参数

| 参数 | 用途 |
| --- | --- |
| --no-voice | 禁用语音交互，保留检测和独立蜂鸣报警 |
| --no-llm | 不启动大模型，使用预制模板 |
| --calibrate | 启动时校准个人 EAR 阈值 |
| --debug | 输出检测指标 |
| --offline-stt | 强制 Vosk 离线识别（默认路径） |
| --online-stt | Google 在线识别，失败后回退 Vosk |

摄像头索引、阈值和模型路径在 config.py 中调整。diagnose_mic.py 和 diagnose_stt.py 用于麦克风及离线语音诊断；前者会录制 5 秒音频到 mic_test.wav。

## 结构

~~~text
main.py                 主程序与视频循环
config.py               检测、语音和模型配置
src/                    检测、指标、疲劳分析、语音、对话与日志模块
diagnose_mic.py          麦克风诊断
diagnose_stt.py          离线语音识别诊断
requirements.txt        Python 依赖
~~~

## 数据与资源

仅发布本项目源码及说明，不包含虚拟环境、模型权重、录音、日志、音乐、论文附件或同目录其他项目。模型需自行下载并遵守各自条款；MIT 许可证不覆盖第三方模型或媒体。

音乐代码保留本地文件名 M800001U9RTs08qoj9.mp3，但不分发音乐。此功能需自行提供有权使用的音频，或修改 src/co_pilot_brain.py 与 src/startup_check.py 中的路径；相关对话模板仍含演示歌曲名称。缺失音乐时自检会提示。

会话日志可能包含识别文字和对话内容，保存在本地 logs/。录音、日志和模型目录均排除在 Git 提交范围之外。启用 --online-stt 会向 Google 识别服务发送语音；默认离线识别不走该服务。

## 使用限制

仅供学习、研究和静止场景演示，不是经过验证的车载安全系统。不能依赖本项目判断是否适合驾驶或代替休息。评分和阈值属于实验性实现，真实道路场景的准确率与可靠性尚未验证。

## 许可证

项目源码使用 [MIT License](LICENSE)。
