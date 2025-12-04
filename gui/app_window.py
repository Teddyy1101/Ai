import cv2
import numpy as np
import os
from tkinter import Tk, Button, Frame, Label, filedialog, messagebox, Canvas
from tkinter import ttk
from core.camera import Camera
from core.detector import Detector
from core.drowsiness_logic import DrowsinessDetector, stop_alarm
from utils.draw_utils import draw_landmarks, draw_face_box, draw_mouth_status
from gui.video_panel import VideoPanel


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Phát hiện Buồn ngủ")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')

        # === HEADER ===
        header = Frame(root, bg="#16213e", height=80)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title = Label(
            header,
            text="HỆ THỐNG PHÁT HIỆN BUỒN NGỦ NGƯỜI LÁI XE",
            font=("Segoe UI", 22, "bold"),
            fg="#00d9ff",
            bg="#16213e"
        )
        title.pack(pady=20)

        # === MAIN CONTAINER ===
        main_container = Frame(root, bg="#1a1a2e")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # === LEFT PANEL - VIDEO ===
        left_panel = Frame(main_container, bg="#0f3460", relief="flat")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        video_title = Label(
            left_panel,
            text="VIDEO GIÁM SÁT",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg="#0f3460",
            pady=10
        )
        video_title.pack()

        self.panel = VideoPanel(left_panel)
        self.panel.pack(padx=10, pady=10)

        # === RIGHT PANEL - STATS ===
        right_panel = Frame(main_container, bg="#0f3460", width=350)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)

        stats_title = Label(
            right_panel,
            text="THÔNG SỐ GIÁM SÁT",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg="#0f3460",
            pady=10
        )
        stats_title.pack()

        # Stats cards container
        stats_container = Frame(right_panel, bg="#0f3460")
        stats_container.pack(fill="both", expand=True, padx=15, pady=10)

        # EAR Card
        self.create_stat_card(
            stats_container,
            "CHỈ SỐ EAR",
            "ear_value",
            "ear_status",
            "#00d9ff",
            0
        )

        # Yawn Card
        self.create_stat_card(
            stats_container,
            "SỐ LẦN NGÁP",
            "yawn_value",
            "yawn_status",
            "#ff6b6b",
            1
        )

        # Alert Card
        alert_card = Frame(stats_container, bg="#1a1a2e", relief="flat", bd=0)
        alert_card.pack(fill="x", pady=15)

        Label(
            alert_card,
            text="TRẠNG THÁI",
            font=("Segoe UI", 11, "bold"),
            fg="#888888",
            bg="#1a1a2e"
        ).pack(pady=(10, 5))

        self.alert_canvas = Canvas(alert_card, width=280, height=100, bg="#1a1a2e", highlightthickness=0)
        self.alert_canvas.pack(pady=10)

        self.alert_label = Label(
            alert_card,
            text="BÌNH THƯỜNG",
            font=("Segoe UI", 20, "bold"),
            fg="#4ecca3",
            bg="#1a1a2e"
        )
        self.alert_label.pack()

        self.counter_label = Label(
            alert_card,
            text="Bộ đếm: 0 / 0",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#1a1a2e"
        )
        self.counter_label.pack(pady=(5, 10))

        # === BOTTOM - CONTROLS ===
        bottom_frame = Frame(root, bg="#16213e", height=100)
        bottom_frame.pack(fill="x", side="bottom")
        bottom_frame.pack_propagate(False)

        # Status bar
        self.status_label = Label(
            bottom_frame,
            text="Sẵn sàng",
            font=("Segoe UI", 12),
            fg="#ffffff",
            bg="#16213e",
            anchor="w",
            padx=20
        )
        self.status_label.pack(fill="x", pady=(10, 5))

        # Control buttons
        btn_container = Frame(bottom_frame, bg="#16213e")
        btn_container.pack(pady=10)

        buttons = [
            ("Bật Camera", self.start_camera, "#4ecca3"),
            ("Mở Video", self.open_video, "#00d9ff"),
            ("Dừng", self.stop, "#ff6b6b"),
            ("Tắt Cảnh Báo", self.mute_alarm, "#ffa500"),
            ("Thoát", self.exit, "#888888")
        ]

        for text, command, color in buttons:
            self.create_modern_button(btn_container, text, command, color)

        # === COMPONENTS ===
        self.running = False
        self.camera = None
        self.detector = Detector()
        self.drowsiness = DrowsinessDetector(mouth_model_path="model/yawn_model.pt")
        self.yawn_count = 0

        # Initial animation
        self.animate_alert_idle()

        print("[INFO] Ứng dụng khởi động thành công")

    def create_stat_card(self, parent, title, value_attr, status_attr, color, position):
        """Tạo card hiển thị thông số"""
        card = Frame(parent, bg="#1a1a2e", relief="flat", bd=0)
        card.pack(fill="x", pady=15)

        Label(
            card,
            text=title,
            font=("Segoe UI", 11, "bold"),
            fg="#888888",
            bg="#1a1a2e"
        ).pack(pady=(10, 5))

        value_label = Label(
            card,
            text="0.00" if "ear" in value_attr else "0",
            font=("Segoe UI", 36, "bold"),
            fg=color,
            bg="#1a1a2e"
        )
        value_label.pack()
        setattr(self, value_attr, value_label)

        status_label = Label(
            card,
            text="Đang chờ...",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#1a1a2e"
        )
        status_label.pack(pady=(0, 10))
        setattr(self, status_attr, status_label)

    def create_modern_button(self, parent, text, command, color):
        """Tạo nút bấm hiện đại"""
        btn = Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg=color,
            activebackground=color,
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=10
        )
        btn.pack(side="left", padx=8)

        # Hover effect
        def on_enter(e):
            btn.config(bg=self.lighten_color(color))

        def on_leave(e):
            btn.config(bg=color)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def lighten_color(self, color):
        """Làm sáng màu khi hover"""
        colors = {
            "#4ecca3": "#6effc3",
            "#00d9ff": "#33e3ff",
            "#ff6b6b": "#ff8888",
            "#ffa500": "#ffb733",
            "#888888": "#aaaaaa"
        }
        return colors.get(color, color)

    def animate_alert_idle(self):
        """Animation khi idle"""
        if not self.running:
            self.alert_canvas.delete("all")
            # Vẽ vòng tròn pulse
            self.alert_canvas.create_oval(90, 20, 190, 120, outline="#4ecca3", width=3)
            self.root.after(1000, self.animate_alert_idle)

    def animate_alert_danger(self):
        """Animation khi nguy hiểm"""
        if self.running and hasattr(self, 'drowsiness') and self.drowsiness.drowsy:
            self.alert_canvas.delete("all")
            # Vẽ tam giác cảnh báo
            self.alert_canvas.create_polygon(
                140, 30, 100, 100, 180, 100,
                fill="#ff6b6b", outline="#ff6b6b"
            )
            self.alert_canvas.create_text(
                140, 75, text="!", font=("Arial", 40, "bold"), fill="#ffffff"
            )
            self.root.after(500, self.animate_alert_danger)

    # =========================================================
    # =============== CHỨC NĂNG ĐIỀU KHIỂN ===================
    # =========================================================
    def start_camera(self):
        """Khởi động camera"""
        self.stop()
        self.camera = Camera(0)
        self.running = True
        self.update_frame()
        self.status_label.config(text="▶ Camera đang hoạt động")
        print("[INFO] Camera đã bật")

    def open_video(self):
        """Mở file video"""
        video_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả", "*.*")]
        )
        if not video_path:
            return

        if not os.path.exists(video_path):
            messagebox.showerror("Lỗi", "Không tìm thấy video!")
            return

        self.stop()
        self.camera = Camera(video_path)
        self.running = True
        self.update_frame()
        self.status_label.config(text="▶ Đang phát video")
        print(f"[INFO] Đã mở video: {video_path}")

    def stop(self):
        """Dừng camera/video"""
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.status_label.config(text="⏸ Đã dừng")
        self.reset_display()
        print("[INFO] Đã dừng")

    def exit(self):
        """Thoát"""
        self.stop()
        self.root.destroy()
        print("[INFO] Đã thoát")

    def mute_alarm(self):
        """Tắt cảnh báo"""
        stop_alarm()
        self.drowsiness.drowsy = False
        self.drowsiness.counter_eye = 0
        self.drowsiness.counter_mouth = 0
        self.drowsiness.yawn_count = 0
        self.drowsiness.last_yawn_state = False
        self.yawn_count = 0

        self.status_label.config(text="🔇 Đã tắt cảnh báo")
        self.alert_label.config(text="BÌNH THƯỜNG", fg="#4ecca3")
        self.counter_label.config(text="Bộ đếm: 0 / 0")
        self.alert_canvas.delete("all")
        print("[INFO] Đã tắt cảnh báo")

    def reset_display(self):
        """Reset hiển thị"""
        self.ear_value.config(text="0.00")
        self.ear_status.config(text="Đang chờ...")
        self.yawn_value.config(text="0")
        self.yawn_status.config(text="Đang chờ...")
        self.alert_label.config(text="BÌNH THƯỜNG", fg="#4ecca3")
        self.counter_label.config(text="Bộ đếm: 0 / 0")
        self.alert_canvas.delete("all")
        self.animate_alert_idle()

    # =========================================================
    # =============== CẬP NHẬT KHUNG HÌNH =====================
    # =========================================================
    def update_frame(self):
        """Xử lý và hiển thị khung hình"""
        if not self.running or not self.camera:
            return

        frame = self.camera.get_frame()
        if frame is None:
            if self.camera.is_video_file:
                self.status_label.config(text="⏹ Video đã kết thúc")
                self.running = False
                stop_alarm()
                self.camera.release()
            return

        faces = self.detector.detect_faces(frame)

        if len(faces) == 0:
            self.status_label.config(text="⚠ Không phát hiện khuôn mặt")
        else:
            for face in faces:
                landmarks = face["landmarks"]
                left_eye = landmarks[self.detector.LEFT_EYE]
                right_eye = landmarks[self.detector.RIGHT_EYE]

                result = self.drowsiness.update(frame, left_eye, right_eye, landmarks)
                ear = result["ear"]
                mouth_class = result["mouth_class"]
                mouth_prob = result["mouth_prob"]
                drowsy = result["is_drowsy"]
                self.yawn_count = result["yawn_count"]

                # Update stats
                self.ear_value.config(text=f"{ear:.3f}")
                if ear < 0.2:
                    self.ear_status.config(text="⚠ Mắt đang nhắm", fg="#ff6b6b")
                else:
                    self.ear_status.config(text="✓ Mắt đang mở", fg="#4ecca3")

                self.yawn_value.config(text=str(self.yawn_count))
                if mouth_class == "yawn":
                    self.yawn_status.config(text=f"⚠ Đang ngáp ({mouth_prob:.0%})", fg="#ff6b6b")
                else:
                    self.yawn_status.config(text=f"✓ Bình thường ({mouth_prob:.0%})", fg="#4ecca3")

                if drowsy:
                    self.alert_label.config(text="NGUY HIỂM!", fg="#ff6b6b")
                    self.status_label.config(text="🚨 CẢNH BÁO: Phát hiện buồn ngủ!")
                    self.animate_alert_danger()
                else:
                    self.alert_label.config(text="BÌNH THƯỜNG", fg="#4ecca3")
                    self.status_label.config(text="✓ Đang giám sát - Tỉnh táo")

                self.counter_label.config(
                    text=f"Bộ đếm: {self.drowsiness.counter_eye} / {self.drowsiness.counter_mouth}"
                )

                # Draw on frame
                draw_face_box(frame, face["rect"], drowsy)
                draw_landmarks(frame, landmarks, self.detector.LEFT_EYE, self.detector.RIGHT_EYE, list(range(48, 68)))
                draw_mouth_status(frame, landmarks, mouth_class, mouth_prob)

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Ngap: {self.yawn_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if drowsy:
                    cv2.putText(frame, "CANH BAO!!!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.panel.update_frame(frame)

        self.root.after(10, self.update_frame)