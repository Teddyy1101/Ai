import dlib
import cv2
import numpy as np
import os

class Detector:
    def __init__(self):
        model_path = os.path.join("assets", "models", "shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

        # Chỉ số điểm mốc mắt trái/phải
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        faces = []
        for rect in rects:
            shape = self.predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=int)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            faces.append({
                "rect": rect,
                "landmarks": coords
            })
        return faces
