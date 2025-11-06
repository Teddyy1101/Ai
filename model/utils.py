import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# 1️⃣ Kiến trúc mô hình giống hệt khi bạn train trên Colab
# =============================
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
            nn.Linear(512, 2)  # 2 lớp: normal và yawn
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# =============================
# 2️⃣ Lớp load mô hình và dự đoán
# =============================
class MouthModel:
    def __init__(self, model_path=None, img_size=(48, 48), grayscale=True, device=None):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_path is None:
            model_path = os.path.join(base, "model", "yawn_model2.pt")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ======= Load model =======
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch model not found: {model_path}")

        try:
            self.model = YawnModel().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[INFO] Loaded PyTorch model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load PyTorch model: {e}")

        # ======= Không dùng class_indices.json =======
        self.idx2label = ["normal", "yawn"]

        self.img_size = img_size
        self.grayscale = grayscale

    def predict(self, roi_bgr):
        """Dự đoán vùng miệng bằng mô hình PyTorch"""
        if roi_bgr is None or roi_bgr.size == 0:
            return None, 0.0, None

        # xử lý ảnh
        if self.grayscale:
            roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, self.img_size)
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)  # (1, H, W)
        else:
            roi = cv2.resize(roi_bgr, self.img_size)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi.transpose(2, 0, 1).astype("float32") / 255.0  # (3, H, W)

        # chuyển sang tensor
        x = torch.from_numpy(roi).unsqueeze(0).to(self.device)  # (1, C, H, W)
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        label = self.idx2label[idx]
        conf = float(probs[idx])
        return label, conf, probs
