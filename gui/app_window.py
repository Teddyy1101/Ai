import cv2
from tkinter import Tk, Button, Frame
from core.camera import Camera
from core.detector import Detector
from core.drowsiness_logic import DrowsinessDetector
from core.alert import play_alarm
from utils.draw_utils import draw_landmarks, draw_face_box
from gui.video_panel import VideoPanel

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection (EAR-based)")
        self.root.geometry("800x600")

        self.panel = VideoPanel(root)
        self.panel.pack()

        btn_frame = Frame(root)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Start", command=self.start).pack(side="left", padx=10)
        Button(btn_frame, text="Stop", command=self.stop).pack(side="left", padx=10)
        Button(btn_frame, text="Exit", command=self.exit).pack(side="left", padx=10)

        self.running = False
        self.camera = None
        self.detector = Detector()
        self.drowsiness = DrowsinessDetector()

    def start(self):
        if not self.running:
            self.camera = Camera(0)
            self.running = True
            self.update_frame()

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()

    def exit(self):
        self.stop()
        self.root.destroy()

    def update_frame(self):
        if self.running:
            frame = self.camera.get_frame()
            if frame is not None:
                faces = self.detector.detect_faces(frame)

                for face in faces:
                    landmarks = face["landmarks"]
                    left_eye = landmarks[self.detector.LEFT_EYE]
                    right_eye = landmarks[self.detector.RIGHT_EYE]

                    ear, drowsy = self.drowsiness.update(left_eye, right_eye)
                    draw_face_box(frame, face["rect"], drowsy)
                    draw_landmarks(frame, landmarks, self.detector.LEFT_EYE, self.detector.RIGHT_EYE)

                    text = f"EAR: {ear:.2f}"
                    cv2.putText(frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if drowsy:
                        cv2.putText(frame, "DROWSY!!!", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        play_alarm()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.panel.update_frame(frame)

            self.root.after(10, self.update_frame)
