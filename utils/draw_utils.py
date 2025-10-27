import cv2

def draw_landmarks(frame, landmarks, left_eye_idx, right_eye_idx):
    """Vẽ hai mắt"""
    for (x, y) in landmarks[left_eye_idx + right_eye_idx]:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def draw_face_box(frame, rect, drowsy=False):
    """Vẽ khung khuôn mặt"""
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    color = (0, 0, 255) if drowsy else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
