from pynput import keyboard  # 键盘监听库

# 初始化键盘状态跟踪字典
key_states = {
    keyboard.Key.up: False,  # 上方向键
    keyboard.Key.down: False,  # 下方向键
    keyboard.Key.left: False,  # 左方向键
    keyboard.Key.right: False,  # 右方向键
    keyboard.Key.alt_l: False,  # 左Alt键（控制Z轴+）
    keyboard.Key.alt_r: False  # 右Alt键（控制Z轴-）
}


# 键盘按下回调函数
def on_press(key):
    if key in key_states:
        key_states[key] = True  # 标记对应按键为激活状态


# 键盘释放回调函数
def on_release(key):
    if key in key_states:
        key_states[key] = False  # 重置按键状态


# 启动键盘监听线程
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()