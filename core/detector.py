# detector.py
import dlib
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# ==========================
#  MÔ HÌNH CNN (PHẢI GIỐNG MÔ HÌNH ĐÃ TRAIN)
# ==========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================
#  LỚP DETECTOR CHÍNH
# ==========================
class Detector:
    def __init__(self):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # ===== Dlib landmarks model =====
        model_path = os.path.join(BASE_DIR, "assets", "models", "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Dlib landmark model not found: {model_path}")

        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load dlib predictor: {e}")

        # ===== Landmarks indices =====
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))

        # ===== Load PyTorch model =====
        cnn_path = os.path.join(BASE_DIR, "model", "yawn_model2.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mouth_model = SimpleCNN().to(self.device)

        if os.path.exists(cnn_path):
            try:
                self.mouth_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
                self.mouth_model.eval()
                print("[INFO] PyTorch mouth model loaded successfully.")
            except Exception as e:
                print(f"[WARN] Failed to load PyTorch model: {e}")
                self.mouth_model = None
        else:
            print(f"[WARN] yawn_model.pt not found at: {cnn_path}")
            self.mouth_model = None

        # Transform cho input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # Mapping class
        self.class_labels = {0: "normal", 1: "yawn"}

    # =====================
    #  PHÁT HIỆN KHUÔN MẶT
    # =====================
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        faces = []

        for rect in rects:
            shape = self.predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=int)
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            face = {"rect": rect, "landmarks": coords}

            if self.mouth_model is not None:
                mouth_status = self._predict_mouth(frame, coords)
                face["mouth_status"] = mouth_status
            else:
                face["mouth_status"] = "unknown"

            faces.append(face)

        return faces

    # =====================
    #  DỰ ĐOÁN VÙNG MIỆNG
    # =====================
    def _predict_mouth(self, frame, landmarks):
        (x_min, y_min) = np.min(landmarks[self.MOUTH], axis=0)
        (x_max, y_max) = np.max(landmarks[self.MOUTH], axis=0)

        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        mouth_roi = frame[y_min:y_max, x_min:x_max]
        if mouth_roi.size == 0:
            return "unknown"

        try:
            img = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.mouth_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                prob_yawn = probs[0][1].item()
                prob_normal = probs[0][0].item()

            if prob_yawn > prob_normal and prob_yawn > 0.8:
                return "yawn"
            else:
                return "normal"

        except Exception as e:
            print(f"[WARN] Mouth prediction failed: {e}")
            return "unknown"
