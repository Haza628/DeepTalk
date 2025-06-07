import sounddevice as sd
import numpy as np
import soundfile as sf
import keyboard  # 监听键盘输入
from faster_whisper import WhisperModel
import ollama
import threading  # 用于异步处理AI响应

# 配置参数
fs = 44100  # 采样率
channels = 1  # 单声道
device = None  # 默认麦克风
output_file = "recording.wav"  # 输出文件名

# 全局变量
is_recording = False
is_ai_responding = False  # 标记AI是否正在生成回复
should_stop_ai = False  # 标记是否应该中断AI响应
audio_buffer = np.array([], dtype=np.float32).reshape(0, channels)  # 存储录音数据
stream = None

# 回调函数：实时捕获音频数据
def callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"音频流错误: {status}")
    if is_recording:
        audio_buffer = np.vstack([audio_buffer, indata.copy()])  # 追加数据

# 开始/停止录音
def toggle_recording():
    global is_recording, stream, audio_buffer, is_ai_responding
    if not is_recording and not is_ai_responding:
        # 开始录音
        print("\n录音开始... (按空格键停止)")
        audio_buffer = np.array([], dtype=np.float32).reshape(0, channels)  # 清空缓冲区
        is_recording = True
        stream = sd.InputStream(
            samplerate=fs,
            channels=channels,
            device=device,
            callback=callback
        )
        stream.start()
    elif is_recording:
        # 停止录音
        print("\n录音停止")
        is_recording = False
        if stream:
            stream.stop()
            stream.close()
        if len(audio_buffer) > 0:
            sf.write(output_file, audio_buffer, fs)  # 保存文件
            print(f"音频已保存到: {output_file}")

            segments, info = whisper_model.transcribe(output_file, beam_size=5)
            input_str = ''
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                input_str += segment.text
            # input_str = '介绍一下sci通信协议'  
            # 添加到历史
            conversation_history.append({"role": "user", "content": input_str})
            
            # 启动AI响应线程
            ai_thread = threading.Thread(target=get_ai_response, args=(conversation_history,))
            ai_thread.start()
        else:
            print("未录制到有效音频")

# 获取AI回复（在独立线程中运行）
def get_ai_response(history):
    global is_ai_responding, should_stop_ai
    is_ai_responding = True
    should_stop_ai = False
    full_response = ""
    
    try:
        response = ollama.chat(model='qwen2.5-coder:7b',   stream=True, messages=history)
        for chunk in response:
            if should_stop_ai:  # 如果按ESC，中断AI响应
                print("\n\n[已中断AI响应] 按空格键重新录音")
                break
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_response += content
    except Exception as e:
        print(f"\nAI响应出错: {e}")
    
    # 如果未被中断，将AI回复添加到历史
    if not should_stop_ai and full_response:
        conversation_history.append({"role": "assistant", "content": full_response})
    
    is_ai_responding = False

# 中断AI响应
def interrupt_ai():
    global should_stop_ai
    if is_ai_responding:
        should_stop_ai = True
        print("\n[正在中断AI响应...]")

# 主程序
model_size = r"models\faster-whisper-medium"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# 初始化对话历史
conversation_history = [
    {"role": "system", "content": '你是一个面试助手，我会给你面试官问我的问题，你帮我回答。回答的内容需要全面。尽量大幅减少think的过程'},
]

print("=== GHZ小助手 ===")
print("按空格键开始/停止录音")
print("按ESC键中断AI响应")

# 监听按键
keyboard.add_hotkey('space', toggle_recording)
keyboard.add_hotkey('esc', interrupt_ai)

# 阻塞主线程，直到程序结束
keyboard.wait('q')  # 这里可以改成其他退出方式，比如 `while True: pass`        