import cv2


class Camera:
    def __init__(self, source=0):
        """
        source = 0 -> webcam
        source = 'path/to/video.mp4' -> video file
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở nguồn video/camera: {source}")

        self.source = source
        self.is_video_file = isinstance(source, str)
        self.is_running = True

    def get_frame(self):
        """Đọc một khung hình từ camera hoặc video"""
        if not self.is_running:
            return None

        ret, frame = self.cap.read()
        if not ret:
            # Nếu là video, khi đọc hết file thì dừng luôn
            if self.is_video_file:
                self.is_running = False
            return None
        return frame

    def release(self):
        """Giải phóng tài nguyên"""
        if self.cap.isOpened():
            self.cap.release()
        self.is_running = False
