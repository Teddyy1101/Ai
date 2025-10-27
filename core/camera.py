import cv2

class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Không thể mở camera!")

    def get_frame(self):
        """Đọc một khung hình từ camera"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Giải phóng tài nguyên camera"""
        if self.cap.isOpened():
            self.cap.release()
