import cv2
import os
import glob
from PIL import Image
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# =============================
# 1️⃣ TRÍCH FRAME TỪ VIDEO
# =============================
def extract_frames(video_path, output_dir, label, step=5):
    """Trích frame từ video vào thư mục label (mỗi step frame lấy 1 ảnh)."""
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:  # lấy mỗi 5 frame để giảm trùng lặp
            filename = f"{label}_{os.path.basename(video_path).split('.')[0]}_{count}.jpg"
            cv2.imwrite(os.path.join(output_dir, label, filename), frame)
        count += 1
    cap.release()

def extract_all_videos(yawdd_path, output_dir):
    print("=== Bắt đầu trích frame từ video... ===")
    for video in glob.glob(f"{yawdd_path}/**/*.avi", recursive=True):
        if "yawn" in video.lower():
            extract_frames(video, output_dir, "yawn")
        else:
            extract_frames(video, output_dir, "normal")
    print("✅ Hoàn tất trích frame.")

# =============================
# 2️⃣ RESIZE ẢNH SAU KHI TRÍCH
# =============================
def resize_images(folder, size=(128, 128)):
    print("=== Đang resize ảnh... ===")
    for subfolder in ["yawn", "normal"]:
        path = os.path.join(folder, subfolder)
        if not os.path.exists(path):
            continue
        for img_path in glob.glob(f"{path}/*.jpg"):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size)
                img.save(img_path)
            except Exception as e:
                print(f"Lỗi ảnh {img_path}: {e}")
    print("✅ Hoàn tất resize ảnh.")

# =============================
# 3️⃣ TẠO FILE LABELS.CSV
# =============================
def create_labels_csv(base_dir, output_csv="dataset/labels.csv"):
    print("=== Đang tạo labels.csv... ===")
    data = []
    for label in ["yawn", "normal"]:
        folder = os.path.join(base_dir, label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(".jpg"):
                data.append({"path": os.path.join(folder, file), "label": label})
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Đã lưu file labels: {output_csv}")
    print(df['label'].value_counts())
    return df

# =============================
# 4️⃣ CÂN BẰNG DỮ LIỆU
# =============================
def balance_dataset(base_dir, method="auto"):
    """
    Cân bằng 2 lớp yawn/normal:
    - Nếu normal > yawn * 2 → Augment yawn (oversample)
    - Nếu normal < yawn → Giữ nguyên
    - Nếu normal gần bằng yawn → Không cần làm gì
    """
    path_yawn = os.path.join(base_dir, "yawn")
    path_normal = os.path.join(base_dir, "normal")
    n_yawn = len(os.listdir(path_yawn))
    n_normal = len(os.listdir(path_normal))
    print(f"📊 Trước khi cân bằng: yawn={n_yawn}, normal={n_normal}")

    # Nếu chênh lệch lớn -> augment
    if n_normal > n_yawn * 1.5:
        print("⚙️ Đang oversample lớp 'yawn' bằng ImageDataGenerator...")
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2]
        )

        images = [f for f in os.listdir(path_yawn) if f.lower().endswith(".jpg")]
        target_num = n_normal - n_yawn
        generated = 0
        i = 0

        while generated < target_num:
            img_name = images[i % len(images)]
            img_path = os.path.join(path_yawn, img_name)
            img = Image.open(img_path).convert("RGB")
            x = np.expand_dims(np.array(img), axis=0)
            for batch in datagen.flow(x, batch_size=1, save_to_dir=path_yawn, save_prefix="aug", save_format="jpg"):
                generated += 1
                if generated >= target_num:
                    break
            i += 1
        print(f"✅ Đã augment thêm {generated} ảnh cho lớp yawn.")
    elif n_yawn > n_normal * 1.5:
        print("⚙️ Đang undersample lớp 'yawn' để cân bằng...")
        all_imgs = [os.path.join(path_yawn, f) for f in os.listdir(path_yawn)]
        keep_imgs = random.sample(all_imgs, n_normal)
        remove_imgs = set(all_imgs) - set(keep_imgs)
        for img in remove_imgs:
            os.remove(img)
        print(f"✅ Đã xóa bớt {len(remove_imgs)} ảnh dư ở lớp yawn.")

    else:
        print("✅ Dataset đã khá cân bằng, bỏ qua bước này.")

# =============================
# 5️⃣ CHẠY TOÀN BỘ QUY TRÌNH
# =============================
if __name__ == "__main__":
    yawdd_path = "Mirror"
    output_dir = "dataset/mouth"

    # Đảm bảo các thư mục con tồn tại
    os.makedirs(os.path.join(output_dir, "yawn"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "normal"), exist_ok=True)

    extract_all_videos(yawdd_path, output_dir)
    resize_images(output_dir, size=(128, 128))
    balance_dataset(output_dir)
    create_labels_csv(output_dir, output_csv="dataset/labels.csv")

    print("🎉 Toàn bộ quá trình chuẩn bị dataset hoàn tất!")

