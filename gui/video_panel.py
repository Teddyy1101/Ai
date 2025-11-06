from tkinter import Label
from PIL import Image, ImageTk

class VideoPanel(Label):
    def __init__(self, parent):
        super().__init__(parent)
        self.image_tk = None

    def update_frame(self, frame):
        """Cập nhật ảnh hiển thị"""
        image = Image.fromarray(frame)
        image = image.resize((800, 480))  # cố định khung hình
        self.image_tk = ImageTk.PhotoImage(image=image)
        self.config(image=self.image_tk)
        self.image_tk.image = self.image_tk
