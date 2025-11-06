import cv2
import numpy as np
import os
from tkinter import Tk, Button, Frame, Label, filedialog, messagebox
from core.camera import Camera
from core.detector import Detector
from core.drowsiness_logic import DrowsinessDetector, stop_alarm
from utils.draw_utils import draw_landmarks, draw_face_box, draw_mouth_status
from gui.video_panel import VideoPanel


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection (EAR + Yawn CNN)")
        self.root.geometry("900x700")

        # --- Video hiển thị ---
        self.panel = VideoPanel(root)
        self.panel.pack()

        # --- Nút điều khiển ---
        btn_frame = Frame(root)
        btn_frame.pack(pady=10)

        Button(btn_frame, text="Start Camera", command=self.start_camera).pack(side="left", padx=10)
        Button(btn_frame, text="Open Video", command=self.open_video).pack(side="left", padx=10)
        Button(btn_frame, text="Stop", command=self.stop).pack(side="left", padx=10)
        Button(btn_frame, text="Mute Alarm", command=self.mute_alarm).pack(side="left", padx=10)  # ✅ Nút mới
        Button(btn_frame, text="Exit", command=self.exit).pack(side="left", padx=10)

        # --- Label hiển thị trạng thái ---
        self.status_label = Label(root, text="Status: Waiting...", font=("Arial", 14))
        self.status_label.pack(pady=5)

        # --- Thành phần chính ---
        self.running = False
        self.camera = None
        self.detector = Detector()
        self.drowsiness = DrowsinessDetector(mouth_model_path="model/yawn_model.pt")

        # --- Bộ đếm ngáp ---
        self.yawn_count = 0

        print("[INFO] App initialized successfully.")

    # =========================================================
    # =============== CHỨC NĂNG CHỌN NGUỒN VIDEO ==============
    # =========================================================
    def start_camera(self):
        """Khởi động camera"""
        self.stop()
        self.camera = Camera(0)
        self.running = True
        self.update_frame()
        print("[INFO] Camera started.")

    def open_video(self):
        """Mở file video từ máy"""
        video_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not video_path:
            return

        if not os.path.exists(video_path):
            messagebox.showerror("Lỗi", "Không tìm thấy video đã chọn!")
            return

        self.stop()
        self.camera = Camera(video_path)
        self.running = True
        self.update_frame()
        print(f"[INFO] Video loaded: {video_path}")

    def stop(self):
        """Dừng camera hoặc video"""
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("[INFO] Stopped video/camera.")

    def exit(self):
        """Thoát chương trình"""
        self.stop()
        self.root.destroy()
        print("[INFO] Application exited.")

    # =========================================================
    # =============== NÚT TẮT CẢNH BÁO ========================
    # =========================================================
    def mute_alarm(self):
        """Tắt âm thanh và reset các thông số"""
        stop_alarm()  # Tắt âm cảnh báo
        # Reset trạng thái buồn ngủ và đếm ngáp
        self.drowsiness.drowsy = False
        self.drowsiness.counter_eye = 0
        self.drowsiness.counter_mouth = 0
        self.drowsiness.yawn_count = 0
        self.drowsiness.last_yawn_state = False
        self.yawn_count = 0

        self.status_label.config(text="Status: Reset done. Alarm muted.")
        print("[INFO] Alarm muted and states reset.")

    # =========================================================
    # =============== CẬP NHẬT KHUNG HÌNH =====================
    # =========================================================
    def update_frame(self):
        """Xử lý và hiển thị khung hình"""
        if not self.running or not self.camera:
            return

        frame = self.camera.get_frame()
        if frame is None:
            # Nếu là video file và hết khung hình
            if self.camera.is_video_file:
                self.status_label.config(text="Status: Video đã kết thúc.")
                self.running = False
                stop_alarm()
                self.camera.release()
            return

        faces = self.detector.detect_faces(frame)

        for face in faces:
            landmarks = face["landmarks"]
            left_eye = landmarks[self.detector.LEFT_EYE]
            right_eye = landmarks[self.detector.RIGHT_EYE]

            # --- Gọi hàm nhận diện buồn ngủ ---
            result = self.drowsiness.update(frame, left_eye, right_eye, landmarks)
            ear = result["ear"]
            mouth_class = result["mouth_class"]
            mouth_prob = result["mouth_prob"]
            drowsy = result["is_drowsy"]
            self.yawn_count = result["yawn_count"]

            # --- Vẽ thông tin ---
            draw_face_box(frame, face["rect"], drowsy)
            draw_landmarks(frame, landmarks, self.detector.LEFT_EYE, self.detector.RIGHT_EYE, list(range(48, 68)))
            draw_mouth_status(frame, landmarks, mouth_class, mouth_prob)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Yawns: {self.yawn_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if drowsy:
                cv2.putText(frame, "ALERT!!!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            self.status_label.config(
                text=f"Status: {'Drowsy!' if drowsy else 'Awake'} | Yawns: {self.yawn_count}"
            )

        # --- Hiển thị frame ---
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.panel.update_frame(frame)

        self.root.after(10, self.update_frame)
