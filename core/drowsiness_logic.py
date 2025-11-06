# drowsiness_logic.py
from scipy.spatial import distance as dist
import threading
import pygame
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ===================== ÂM THANH CẢNH BÁO =====================
is_playing = False

def play_alarm():
    global is_playing
    if is_playing:
        return
    sound_path = os.path.join("assets", "sounds", "warning.wav")
    if not os.path.exists(sound_path):
        print(f"[WARN] Alarm sound not found: {sound_path}")
        return

    def _play():
        global is_playing
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play(-1)
            is_playing = True
        except Exception as e:
            print(f"[ERROR] Failed to play alarm: {e}")

    threading.Thread(target=_play, daemon=True).start()


def stop_alarm():
    global is_playing
    if pygame.mixer.get_init() and is_playing:
        pygame.mixer.music.stop()
        is_playing = False


# ===================== MÔ HÌNH PYTORCH (PHÁT HIỆN NGÁP) =====================
class YawnModel(nn.Module):
    def __init__(self):
        super(YawnModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ===================== PHÁT HIỆN BUỒN NGỦ =====================
class DrowsinessDetector:
    def __init__(
        self,
        ear_thresh=0.27,
        ear_consec_frames=25,
        mouth_model_path="model/yawn_model2.pt",
        mouth_open_thresh=0.8,
        mouth_consec_frames=10,
        mar_thresh=0.6,
        alarm_duration=4.0
    ):
        self.EAR_THRESH = ear_thresh
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.counter_eye = 0
        self.counter_mouth = 0
        self.MAR_THRESH = mar_thresh
        self.MOUTH_THRESH = mouth_open_thresh
        self.MOUTH_CONSEC_FRAMES = mouth_consec_frames
        self.drowsy = False
        self.last_yawn_state = False
        self.ALARM_DURATION = alarm_duration
        self.alarm_start_time = 0
        self.yawn_count = 0

        # Load PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mouth_model = YawnModel().to(self.device)
        if os.path.exists(mouth_model_path):
            try:
                self.mouth_model.load_state_dict(torch.load(mouth_model_path, map_location=self.device))
                self.mouth_model.eval()
                print("[INFO] Loaded YawnModel (PyTorch) successfully.")
            except Exception as e:
                print(f"[WARN] Failed to load PyTorch model: {e}")
                self.mouth_model = None
        else:
            print(f"[WARN] PyTorch model not found: {mouth_model_path}")
            self.mouth_model = None

        # Transform (giống khi train)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.idx_to_class = {0: "normal", 1: "yawn"}


    # ===== Tính EAR =====
    def compute_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)


    # ===== Tính MAR =====
    def compute_mar(self, mouth):
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        return (A + B + C) / (3.0 * D)


    # ===== Cắt vùng miệng =====
    def crop_mouth(self, frame, landmarks):
        if landmarks is None or len(landmarks) < 68:
            return None, None
        mouth = landmarks[48:68]
        x, y, w, h = cv2.boundingRect(np.array(mouth))
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        mouth_roi = frame[y:y + h + margin, x:x + w + margin]
        return mouth_roi, mouth


    # ===== Dự đoán ngáp =====
    def classify_mouth(self, mouth_roi, mar):
        if self.mouth_model is None or mouth_roi is None or mouth_roi.size == 0:
            return "normal", 0.0

        if mar < self.MAR_THRESH:
            return "normal", 1.0

        try:
            img = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.mouth_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                prob_yawn = probs[0][1].item()
                prob_normal = probs[0][0].item()
            return ("yawn", prob_yawn) if prob_yawn > self.MOUTH_THRESH else ("normal", prob_normal)
        except Exception as e:
            print(f"[ERROR] classify_mouth failed: {e}")
            return "unknown", 0.0


    # ===== Cập nhật mỗi frame =====
    def update(self, frame, left_eye, right_eye, landmarks):
        ear_left = self.compute_ear(left_eye)
        ear_right = self.compute_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        mouth_roi, mouth_points = self.crop_mouth(frame, landmarks)
        mar = self.compute_mar(mouth_points) if mouth_points is not None else 0
        mouth_class, mouth_prob = self.classify_mouth(mouth_roi, mar)

        # --- Đếm ngáp ---
        if mouth_class == "yawn" and mouth_prob > self.MOUTH_THRESH:
            if not self.last_yawn_state:
                self.yawn_count += 1
                print(f"[INFO] Yawn detected! (MAR={mar:.2f}, Prob={mouth_prob:.2f}) Count={self.yawn_count}")
            self.last_yawn_state = True
        else:
            self.last_yawn_state = False

        # --- Đếm nhắm mắt ---
        if ear < self.EAR_THRESH:
            self.counter_eye += 1
        else:
            self.counter_eye = 0

        # --- Phát cảnh báo ---
        if self.counter_eye >= self.EAR_CONSEC_FRAMES or self.yawn_count >= 3:
            if not self.drowsy:
                play_alarm()
                self.alarm_start_time = time.time()
            self.drowsy = True
        else:
            # Nếu đang cảnh báo thì đợi đủ 3 giây rồi mới dừng
            if self.drowsy:
                elapsed = time.time() - self.alarm_start_time
                if elapsed >= self.ALARM_DURATION:
                    stop_alarm()
                    self.drowsy = False
                else:
                    # vẫn giữ cảnh báo trong thời gian còn lại
                    self.drowsy = True

        return {
            "ear": round(float(ear), 3),
            "mar": round(float(mar), 3),
            "mouth_class": mouth_class,
            "mouth_prob": round(float(mouth_prob), 2),
            "is_drowsy": self.drowsy,
            "yawn_count": self.yawn_count
        }
