# drowsiness_logic.py - Updated with adaptive MAR threshold based on mode
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
            mar_thresh=0.6,  # Ngưỡng MAR cho mode "algorithm"
            mar_thresh_cnn_filter=0.2,  # NEW: Ngưỡng MAR nhỏ để lọc miệng ngậm cho CNN
            alarm_duration=4.0,
            detection_mode="both"  # "cnn", "algorithm", "both"
    ):
        self.EAR_THRESH = ear_thresh
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.counter_eye = 0
        self.counter_mouth = 0

        # NEW: Ngưỡng MAR linh hoạt
        self.MAR_THRESH = mar_thresh  # Ngưỡng lớn cho algorithm/both
        self.MAR_THRESH_CNN_FILTER = mar_thresh_cnn_filter  # Ngưỡng nhỏ để lọc cho CNN

        self.MOUTH_THRESH = mouth_open_thresh
        self.MOUTH_CONSEC_FRAMES = mouth_consec_frames
        self.drowsy = False
        self.last_yawn_state = False
        self.ALARM_DURATION = alarm_duration
        self.alarm_start_time = 0
        self.yawn_count = 0

        # Detection mode
        self.detection_mode = detection_mode

        # FPR Tracking
        self.total_frames = 0
        self.cnn_yawn_frames = 0
        self.mar_yawn_frames = 0

        # Load PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mouth_model = YawnModel().to(self.device)
        if os.path.exists(mouth_model_path):
            try:
                self.mouth_model.load_state_dict(torch.load(mouth_model_path, map_location=self.device))
                self.mouth_model.eval()
                print(f"[INFO] Loaded YawnModel - Mode: {detection_mode}")
                print(f"[INFO] MAR thresholds - Algorithm: {mar_thresh}, CNN filter: {mar_thresh_cnn_filter}")
            except Exception as e:
                print(f"[WARN] Failed to load PyTorch model: {e}")
                self.mouth_model = None
        else:
            print(f"[WARN] PyTorch model not found: {mouth_model_path}")
            self.mouth_model = None

        # Transform
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

    # ===== Phát hiện ngáp bằng CNN (có lọc MAR khi chạy độc lập) =====
    def detect_yawn_cnn(self, mouth_roi, mar, mar_filter_threshold):
        """
        Phát hiện ngáp bằng CNN model
        - mar_filter_threshold: ngưỡng MAR để lọc trước khi cho CNN quyết định
        """
        if self.mouth_model is None or mouth_roi is None or mouth_roi.size == 0:
            return False, 0.0

        # Kiểm tra MAR trước - nếu miệng ngậm quá thì bỏ qua
        if mar < mar_filter_threshold:
            return False, 0.0  # Miệng đang ngậm, chắc chắn không ngáp

        try:
            img = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.mouth_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                prob_yawn = probs[0][1].item()

            is_yawn = prob_yawn > self.MOUTH_THRESH
            return is_yawn, prob_yawn
        except Exception as e:
            print(f"[ERROR] detect_yawn_cnn failed: {e}")
            return False, 0.0

    # ===== Phát hiện ngáp bằng MAR thuần =====
    def detect_yawn_mar(self, mar):
        """
        Phát hiện ngáp bằng thuật toán MAR
        - Dùng ngưỡng MAR_THRESH (cao hơn)
        """
        is_yawn = mar > self.MAR_THRESH
        return is_yawn, mar

    # ===== Cập nhật mỗi frame =====
    def update(self, frame, left_eye, right_eye, landmarks):
        self.total_frames += 1

        ear_left = self.compute_ear(left_eye)
        ear_right = self.compute_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        mouth_roi, mouth_points = self.crop_mouth(frame, landmarks)
        mar = self.compute_mar(mouth_points) if mouth_points is not None else 0

        # Phát hiện ngáp theo mode
        cnn_yawn = False
        cnn_prob = 0.0
        mar_yawn = False
        final_yawn = False
        final_prob = 0.0

        if self.detection_mode == "cnn":
            # Mode CNN độc lập: dùng CNN với MAR filter nhỏ (0.35)
            cnn_yawn, cnn_prob = self.detect_yawn_cnn(mouth_roi, mar, self.MAR_THRESH_CNN_FILTER)
            final_yawn = cnn_yawn
            final_prob = cnn_prob

        elif self.detection_mode == "algorithm":
            # Mode Algorithm: chỉ dùng MAR với ngưỡng cao (0.6)
            mar_yawn, _ = self.detect_yawn_mar(mar)
            final_yawn = mar_yawn
            final_prob = mar

        else:  # both
            # Mode Both: CNN với MAR filter lớn (0.6) + MAR với ngưỡng cao (0.6)
            cnn_yawn, cnn_prob = self.detect_yawn_cnn(mouth_roi, mar, self.MAR_THRESH)
            mar_yawn, _ = self.detect_yawn_mar(mar)
            final_yawn = cnn_yawn or mar_yawn
            final_prob = max(cnn_prob, mar / self.MAR_THRESH if self.MAR_THRESH > 0 else 0)

        # Cập nhật FPR counters
        if cnn_yawn:
            self.cnn_yawn_frames += 1
        if mar_yawn:
            self.mar_yawn_frames += 1

        # Đếm ngáp (chỉ đếm khi chuyển từ không ngáp -> ngáp)
        if final_yawn:
            if not self.last_yawn_state:
                self.yawn_count += 1
                mode_info = ""
                if self.detection_mode == "cnn":
                    mode_info = f" (CNN: {cnn_prob:.2f}, MAR: {mar:.2f})"
                elif self.detection_mode == "algorithm":
                    mode_info = f" (MAR: {mar:.2f})"
                else:  # both
                    mode_info = f" (CNN: {cnn_yawn}, MAR: {mar_yawn}, mar_val: {mar:.2f})"
                print(f"[INFO] Yawn #{self.yawn_count}{mode_info}")
            self.last_yawn_state = True
        else:
            self.last_yawn_state = False

        # Đếm nhắm mắt
        if ear < self.EAR_THRESH:
            self.counter_eye += 1
        else:
            self.counter_eye = 0

        # Phát cảnh báo
        if self.counter_eye >= self.EAR_CONSEC_FRAMES or self.yawn_count >= 3:
            if not self.drowsy:
                play_alarm()
                self.alarm_start_time = time.time()
            self.drowsy = True
        else:
            if self.drowsy:
                elapsed = time.time() - self.alarm_start_time
                if elapsed >= self.ALARM_DURATION:
                    stop_alarm()
                    self.drowsy = False

        # Tính FPR
        fpr_cnn = (self.cnn_yawn_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        fpr_mar = (self.mar_yawn_frames / self.total_frames * 100) if self.total_frames > 0 else 0

        return {
            "ear": round(float(ear), 3),
            "mar": round(float(mar), 3),
            "mouth_class": "yawn" if final_yawn else "normal",
            "mouth_prob": round(float(final_prob), 2),
            "is_drowsy": self.drowsy,
            "yawn_count": self.yawn_count,
            # FPR data
            "cnn_yawn": cnn_yawn,
            "mar_yawn": mar_yawn,
            "fpr_cnn": round(fpr_cnn, 1),
            "fpr_mar": round(fpr_mar, 1),
            "total_frames": self.total_frames,
            "detection_mode": self.detection_mode,
            # NEW: Thêm thông tin ngưỡng đang dùng
            "mar_threshold_used": self.MAR_THRESH_CNN_FILTER if self.detection_mode == "cnn" else self.MAR_THRESH
        }

    def reset_statistics(self):
        """Reset bộ đếm FPR và yawn count"""
        self.total_frames = 0
        self.cnn_yawn_frames = 0
        self.mar_yawn_frames = 0
        self.yawn_count = 0
        self.counter_eye = 0
        self.counter_mouth = 0
        self.last_yawn_state = False
        print("[INFO] Statistics reset")