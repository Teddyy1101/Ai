import threading
import pygame
import os

# Biến trạng thái toàn cục
is_playing = False

def play_alarm():
    """Phát âm thanh cảnh báo bằng pygame (chỉ phát khi chưa phát trước đó)"""
    global is_playing
    if is_playing:  # Nếu đang phát thì bỏ qua
        return

    sound_path = os.path.join("assets", "sounds", "alarm.wav")

    def _play():
        global is_playing
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)  # -1: lặp vô hạn
        is_playing = True

    threading.Thread(target=_play, daemon=True).start()


def stop_alarm():
    """Dừng âm thanh cảnh báo"""
    global is_playing
    if pygame.mixer.get_init() and is_playing:
        pygame.mixer.music.stop()
        is_playing = False
