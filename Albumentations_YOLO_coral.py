import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from multiprocessing import Manager, Lock, freeze_support, cpu_count
import concurrent.futures
import random
import math

# ========= 配置 =========
root_dir = r"E:\YOLO_DataSet\coral-new"  # 可以是父目录或单个珊瑚文件夹
default_num_aug_per_image = 3  # 默认每张图增强几次
target_total_images = 500      # 可选参数：预计总图片数量，如果提供，将动态计算num_aug_per_image 默认为None
max_retry_per_image = 5        # 每张图每次增强最多重试次数

# ========= Albumentations 增强 pipeline =========
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.MedianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.3),
    A.ISONoise(p=0.3),
    A.ElasticTransform(alpha=50, sigma=5, p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
    A.OpticalDistortion(distort_limit=0.2, p=0.2),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05),
             rotate=(-10, 10), shear=(-5, 5), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    fill=0, p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=0.3),
],
    p=1.0,
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0)
)

# ========= 自定义旋转 =========
def rotate_expand(img, bboxes, angle):
    h, w = img.shape[:2]
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    canvas = np.zeros((diag, diag, 3), dtype=img.dtype)

    x_offset = (diag - w) // 2
    y_offset = (diag - h) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img

    center = (diag // 2, diag // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(canvas, M, (diag, diag), borderValue=(0, 0, 0))

    new_bboxes = []
    for (x, y, bw, bh) in bboxes:
        abs_x = x * w + x_offset
        abs_y = y * h + y_offset
        abs_w = bw * w
        abs_h = bh * h

        box_pts = np.array([
            [abs_x - abs_w / 2, abs_y - abs_h / 2],
            [abs_x + abs_w / 2, abs_y - abs_h / 2],
            [abs_x + abs_w / 2, abs_y + abs_h / 2],
            [abs_x - abs_w / 2, abs_y + abs_h / 2]
        ])
        ones = np.ones((4, 1))
        pts_ones = np.hstack([box_pts, ones])
        rotated_pts = M.dot(pts_ones.T).T

        x_min, y_min = rotated_pts[:, 0].min(), rotated_pts[:, 1].min()
        x_max, y_max = rotated_pts[:, 0].max(), rotated_pts[:, 1].max()

        cx = (x_min + x_max) / 2 / diag
        cy = (y_min + y_max) / 2 / diag
        bw = (x_max - x_min) / diag
        bh = (y_max - y_min) / diag

        cx, cy, bw, bh = map(lambda v: min(max(v, 0.0), 1.0), [cx, cy, bw, bh])
        if bw > 0 and bh > 0:
            new_bboxes.append([cx, cy, bw, bh])

    return rotated_img, new_bboxes

# ========= 单张图像处理 =========
def process_image(img_file, labels_dir, output_img_dir, output_lbl_dir, num_aug_per_image, global_discarded, global_total, lock):
    base = os.path.splitext(os.path.basename(img_file))[0]
    label_file = os.path.join(labels_dir, base + ".txt")

    img = cv2.imread(img_file)
    if img is None or not os.path.exists(label_file):
        return 0

    with open(label_file, "r") as f:
        lines = f.readlines()

    bboxes, class_labels = [], []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        x, y, w, h = [min(max(v, 0.0), 1.0) for v in (x, y, w, h)]
        if w > 0 and h > 0:
            bboxes.append([x, y, w, h])
            class_labels.append(cls)

    discarded_count = 0
    for aug_idx in range(num_aug_per_image):
        success = False
        for attempt in range(max_retry_per_image):
            aug_img, aug_bboxes, aug_labels = img.copy(), bboxes[:], class_labels[:]

            if random.random() < 0.75:
                angle = random.uniform(-30, 30)
                aug_img, aug_bboxes = rotate_expand(aug_img, aug_bboxes, angle)

            try:
                transformed = albumentations_transform(image=aug_img, bboxes=aug_bboxes, class_labels=aug_labels)
                aug_img = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_labels = transformed["class_labels"]

                if len(aug_bboxes) == 0:
                    raise ValueError("增强后没有保留 bboxes")
                success = True
                break
            except Exception:
                continue

        if not success:
            discarded_count += len(bboxes)
            continue

        save_img_path = os.path.join(output_img_dir, f"{base}_aug{aug_idx + 1}.jpg")
        save_lbl_path = os.path.join(output_lbl_dir, f"{base}_aug{aug_idx + 1}.txt")
        cv2.imwrite(save_img_path, aug_img)
        with open(save_lbl_path, "w") as f:
            for cls, (x, y, w, h) in zip(aug_labels, aug_bboxes):
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    with lock:
        global_discarded.value += discarded_count
        global_total.value += len(bboxes) * num_aug_per_image

    return discarded_count

# ========= 遍历目录 =========
def get_coral_folders(root_dir):
    if os.path.exists(os.path.join(root_dir, "images")) and os.path.exists(os.path.join(root_dir, "labels")):
        return [root_dir]
    folders = []
    for d in os.listdir(root_dir):
        full_path = os.path.join(root_dir, d)
        if os.path.isdir(full_path) and \
           os.path.exists(os.path.join(full_path, "images")) and \
           os.path.exists(os.path.join(full_path, "labels")):
            folders.append(full_path)
    return folders

# ========= 主程序 =========
if __name__ == "__main__":
    freeze_support()  # Windows 多进程必须
    manager = Manager()
    global_discarded = manager.Value("i", 0)
    global_total = manager.Value("i", 0)
    lock = manager.Lock()

    cpu_cores = max(1, cpu_count() - 2)
    print(f"⚡ 检测到 CPU 核心数: {cpu_count()}，设置 max_workers={cpu_cores}")

    coral_folders = get_coral_folders(root_dir)
    if not coral_folders:
        print("❌ 未找到任何可用的珊瑚文件夹")
        exit(0)

    for folder in coral_folders:
        print(f"📁 处理文件夹: {folder}")
        input_img_dir = os.path.join(folder, "images")
        input_lbl_dir = os.path.join(folder, "labels")

        img_files = [
            os.path.join(input_img_dir, f)
            for f in os.listdir(input_img_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        if len(img_files) == 0:
            print("⚠️ 该文件夹没有可用图像，跳过。")
            continue

        # 动态计算num_aug_per_image
        if target_total_images is not None:
            num_aug_per_image = max(0, math.ceil(target_total_images / len(img_files)) - 1)
            print(f"⚡ 动态设置 num_aug_per_image = {num_aug_per_image} (目标总图像 {target_total_images})")
        else:
            num_aug_per_image = default_num_aug_per_image

        if num_aug_per_image <= 0:
            print(f"✅ 增强次数为0，原始图像数量 {len(img_files)} 已达到预期或增强数量设置为0，不生成 Albumentations_plus")
            continue

        # 生成输出目录
        output_img_dir = os.path.join(folder, "Albumentations_plus", "images")
        output_lbl_dir = os.path.join(folder, "Albumentations_plus", "labels")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)

        # 多进程处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            list(tqdm(executor.map(
                process_image,
                img_files,
                [input_lbl_dir] * len(img_files),
                [output_img_dir] * len(img_files),
                [output_lbl_dir] * len(img_files),
                [num_aug_per_image] * len(img_files),
                [global_discarded] * len(img_files),
                [global_total] * len(img_files),
                [lock] * len(img_files),
            ), total=len(img_files), desc="Augmenting", ncols=100))

    # 全局统计
    if global_total.value > 0:
        ratio = global_discarded.value / global_total.value * 100
        print(f"\n📊 全局统计: 丢弃 {global_discarded.value} / {global_total.value} bboxes ({ratio:.2f}%)")
    else:
        print("\n📊 全局统计: 没有检测到任何 bboxes")
