import cv2
import numpy as np


def draw_landmarks(frame, landmarks, left_eye_idx, right_eye_idx, mouth_idx):
    """Vẽ landmarks cho mắt và miệng"""
    # --- Vẽ mắt ---
    for (x, y) in landmarks[left_eye_idx + right_eye_idx]:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # --- Vẽ miệng ---
    for (x, y) in landmarks[mouth_idx]:
        cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)


def draw_face_box(frame, rect, drowsy=False):
    """Vẽ khung khuôn mặt (màu đỏ nếu buồn ngủ / ngáp)"""
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    color = (0, 0, 255) if drowsy else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)


def draw_mouth_status(frame, landmarks, mouth_class, mouth_prob):
    """Vẽ chấm landmark và text trạng thái miệng (không hiển thị % xác suất)"""
    mouth = landmarks[48:68]

    # Chuyển mouth_class từ số sang nhãn chuỗi
    class_labels = {
        0: "Closed",   # Miệng đóng
        1: "Yawn"      # Miệng mở (ngáp)
    }

    # Nếu mouth_class là số thì ánh xạ sang nhãn
    if not isinstance(mouth_class, str):
        mouth_label = class_labels.get(int(mouth_class), "Unknown")
    else:
        mouth_label = mouth_class

    # Chọn màu theo trạng thái miệng
    color = (0, 0, 255) if mouth_label.lower() == "yawn" else (0, 255, 0)

    # Vẽ chấm landmark quanh miệng
    for (x, y) in mouth:
        cv2.circle(frame, (x, y), 2, color, -1)

    # Tính vị trí hiển thị text
    x, y, w, h = cv2.boundingRect(np.array(mouth))
    label = mouth_label.upper()  # Không còn phần trăm %

    # Vẽ text trên khung hình
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

