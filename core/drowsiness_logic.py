from scipy.spatial import distance as dist
import threading
import pygame
import os


# ===================== ÂM THANH CẢNH BÁO =====================
is_playing = False  # Biến toàn cục để kiểm tra trạng thái âm thanh

def play_alarm():
    """Phát âm thanh cảnh báo bằng pygame (chỉ phát khi chưa phát trước đó)"""
    global is_playing
    if is_playing:  # Nếu đang phát rồi thì không phát thêm
        return

    sound_path = os.path.join("assets", "sounds", "alarm.wav")

    def _play():
        global is_playing
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)  # Lặp vô hạn
        is_playing = True

    threading.Thread(target=_play, daemon=True).start()


def stop_alarm():
    """Dừng âm thanh cảnh báo"""
    global is_playing
    if pygame.mixer.get_init() and is_playing:
        pygame.mixer.music.stop()
        is_playing = False


# ===================== PHÁT HIỆN BUỒN NGỦ =====================
class DrowsinessDetector:
    def __init__(self, ear_thresh=0.23, ear_consec_frames=20):
        """
        ear_thresh: ngưỡng EAR dưới mức này coi là nhắm
        ear_consec_frames: số frame liên tiếp để xác nhận buồn ngủ
        """
        self.EAR_THRESH = ear_thresh
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.counter = 0
        self.drowsy = False

    def compute_ear(self, eye):
        """Tính EAR từ 6 điểm mắt"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def update(self, left_eye, right_eye):
        """Cập nhật trạng thái và điều khiển cảnh báo"""
        ear_left = self.compute_ear(left_eye)
        ear_right = self.compute_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        if ear < self.EAR_THRESH:
            self.counter += 1
            if self.counter >= self.EAR_CONSEC_FRAMES:
                if not self.drowsy:
                    play_alarm()  # Bắt đầu phát cảnh báo
                self.drowsy = True
        else:
            if self.drowsy:
                stop_alarm()  # Dừng cảnh báo khi tỉnh lại
            self.counter = 0
            self.drowsy = False

        return ear, self.drowsy
