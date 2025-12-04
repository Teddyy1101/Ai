# app_window.py - Updated with mode selection and FPR comparison
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
        self.root.title("Hệ thống Phát hiện Buồn ngủ - So sánh CNN & Algorithm")
        self.root.geometry("1400x850")
        self.root.configure(bg="#1a1a2e")

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground='#0f3460', background='#00d9ff',
                        foreground='white', arrowcolor='white')

        # === HEADER ===
        header = Frame(root, bg="#16213e", height=80)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title = Label(
            header,
            text="HỆ THỐNG PHÁT HIỆN BUỒN NGỦ - SO SÁNH CNN & ALGORITHM",
            font=("Segoe UI", 20, "bold"),
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
        right_panel = Frame(main_container, bg="#0f3460", width=450)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)

        stats_title = Label(
            right_panel,
            text="THÔNG SỐ & SO SÁNH",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg="#0f3460",
            pady=10
        )
        stats_title.pack()

        # === NEW: MODE SELECTION ===
        mode_frame = Frame(right_panel, bg="#1a1a2e", relief="flat")
        mode_frame.pack(fill="x", padx=15, pady=10)

        Label(
            mode_frame,
            text="CHẾ ĐỘ PHÁT HIỆN",
            font=("Segoe UI", 11, "bold"),
            fg="#888888",
            bg="#1a1a2e"
        ).pack(pady=(10, 5))

        self.mode_var = ttk.Combobox(
            mode_frame,
            values=["CNN Only", "Algorithm (MAR) Only", "Both (CNN + MAR)"],
            state="readonly",
            font=("Segoe UI", 10),
            width=25
        )
        self.mode_var.set("Both (CNN + MAR)")
        self.mode_var.pack(pady=5)
        self.mode_var.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Stats container
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

        # === NEW: FPR COMPARISON CARD ===
        fpr_card = Frame(stats_container, bg="#1a1a2e", relief="flat", bd=0)
        fpr_card.pack(fill="x", pady=15)

        Label(
            fpr_card,
            text="SO SÁNH FPR (False Positive Rate)",
            font=("Segoe UI", 11, "bold"),
            fg="#888888",
            bg="#1a1a2e"
        ).pack(pady=(10, 5))

        self.fpr_comparison = Label(
            fpr_card,
            text="Chưa có dữ liệu",
            font=("Segoe UI", 10),
            fg="#ffffff",
            bg="#1a1a2e",
            justify="left"
        )
        self.fpr_comparison.pack(pady=5)


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
            ("Reset Stats", self.reset_stats, "#9b59b6"),  # NEW
            ("Thoát", self.exit, "#888888")
        ]

        for text, command, color in buttons:
            self.create_modern_button(btn_container, text, command, color)

        # === COMPONENTS ===
        self.running = False
        self.camera = None
        self.detector = Detector()
        self.drowsiness = None
        self.init_drowsiness_detector()

        # Initial animation
        self.animate_alert_idle()

        print("[INFO] Ứng dụng khởi động thành công")

    def init_drowsiness_detector(self):
        """Khởi tạo detector theo mode được chọn"""
        mode_map = {
            "CNN Only": "cnn",
            "Algorithm (MAR) Only": "algorithm",
            "Both (CNN + MAR)": "both"
        }
        selected_mode = mode_map[self.mode_var.get()]

        self.drowsiness = DrowsinessDetector(
            mouth_model_path="model/yawn_model.pt",
            detection_mode=selected_mode
        )
        print(f"[INFO] Detector initialized with mode: {selected_mode}")

    def on_mode_change(self, event=None):
        """Xử lý khi thay đổi mode"""
        if self.running:
            messagebox.showinfo("Thông báo", "Vui lòng dừng camera/video trước khi thay đổi chế độ!")
            return

        self.init_drowsiness_detector()
        self.status_label.config(text=f"✓ Đã chuyển sang chế độ: {self.mode_var.get()}")

    def reset_stats(self):
        """Reset thống kê FPR"""
        if self.drowsiness:
            self.drowsiness.reset_statistics()
            self.fpr_comparison.config(text="Đã reset - Chưa có dữ liệu")
            self.status_label.config(text="✓ Đã reset thống kê")
            print("[INFO] Statistics reset by user")

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
            "#9b59b6": "#b87fd9",
            "#888888": "#aaaaaa"
        }
        return colors.get(color, color)

    def animate_alert_idle(self):
        """Animation khi idle"""
        if not self.running:
            self.root.after(1000, self.animate_alert_idle)

    def animate_alert_danger(self):
        """Animation khi nguy hiểm"""
        if self.running and hasattr(self, 'drowsiness') and self.drowsiness.drowsy:
            self.root.after(500, self.animate_alert_danger)

    def start_camera(self):
        """Khởi động camera"""
        self.stop()
        self.camera = Camera(0)
        self.running = True
        self.update_frame()
        self.status_label.config(text=f"▶ Camera - Mode: {self.mode_var.get()}")
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
        self.status_label.config(text=f"▶ Video - Mode: {self.mode_var.get()}")
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
        if self.drowsiness:
            self.drowsiness.drowsy = False
            self.drowsiness.counter_eye = 0
            self.drowsiness.last_yawn_state = False

        self.status_label.config(text="🔇 Đã tắt cảnh báo")
        print("[INFO] Đã tắt cảnh báo")

    def reset_display(self):
        """Reset hiển thị"""
        self.ear_value.config(text="0.00")
        self.ear_status.config(text="Đang chờ...")
        self.fpr_comparison.config(text="Chưa có dữ liệu")
        self.animate_alert_idle()

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

                # NEW: FPR data
                fpr_cnn = result["fpr_cnn"]
                fpr_mar = result["fpr_mar"]
                total_frames = result["total_frames"]
                cnn_yawn = result.get("cnn_yawn", False)
                mar_yawn = result.get("mar_yawn", False)
                mode = result["detection_mode"]

                # Update stats
                self.ear_value.config(text=f"{ear:.3f}")
                if ear < 0.2:
                    self.ear_status.config(text="⚠ Mắt đang nhắm", fg="#ff6b6b")
                else:
                    self.ear_status.config(text="✓ Mắt đang mở", fg="#4ecca3")


                # NEW: Update FPR comparison
                if total_frames > 30:  # Chỉ hiển thị sau 30 frames
                    if mode == "both":
                        comparison_text = f"📊 Tổng frames: {total_frames}\n\n"
                        comparison_text += f"CNN: {fpr_cnn}% frames ngáp\n"
                        comparison_text += f"MAR: {fpr_mar}% frames ngáp\n\n"

                        if fpr_cnn > fpr_mar:
                            diff = fpr_cnn - fpr_mar
                            comparison_text += f"⚠ CNN có FPR cao hơn {diff:.1f}%"
                            self.fpr_comparison.config(fg="#ff6b6b")
                        elif fpr_mar > fpr_cnn:
                            diff = fpr_mar - fpr_cnn
                            comparison_text += f"✅ MAR có FPR cao hơn {diff:.1f}%"
                            self.fpr_comparison.config(fg="#ffa500")
                        else:
                            comparison_text += "✅ FPR bằng nhau"
                            self.fpr_comparison.config(fg="#4ecca3")
                    else:
                        comparison_text = f"📊 Tổng frames: {total_frames}\n\n"
                        if mode == "cnn":
                            comparison_text += f"CNN: {fpr_cnn}% frames ngáp"
                        else:
                            comparison_text += f"MAR: {fpr_mar}% frames ngáp"
                        self.fpr_comparison.config(fg="#ffffff")

                    self.fpr_comparison.config(text=comparison_text)

                if drowsy:
                    self.status_label.config(text="🚨 CẢNH BÁO: Phát hiện buồn ngủ!")
                    self.animate_alert_danger()
                else:
                    self.status_label.config(text=f"✓ Giám sát - Mode: {self.mode_var.get()}")

                # Draw on frame
                draw_face_box(frame, face["rect"], drowsy)
                draw_landmarks(frame, landmarks, self.detector.LEFT_EYE, self.detector.RIGHT_EYE, list(range(48, 68)))
                draw_mouth_status(frame, landmarks, mouth_class, mouth_prob)

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Ngap: {self.yawn_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Show mode and detection status
                mode_text = f"Mode: {mode.upper()}"
                if mode == "both":
                    mode_text += f" (C:{int(cnn_yawn)} M:{int(mar_yawn)})"
                cv2.putText(frame, mode_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if drowsy:
                    cv2.putText(frame, "CANH BAO!!!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.panel.update_frame(frame)

        self.root.after(10, self.update_frame)