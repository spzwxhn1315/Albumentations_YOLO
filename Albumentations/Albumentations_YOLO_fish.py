import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from multiprocessing import Manager, Lock, freeze_support, cpu_count
import concurrent.futures
import random
import math
import datetime

# 假设 RandomLightCircle.py 文件存在于同一目录下
from RandomLightCircle import AdvancedRandomLightCircle  # 自定义水下光斑类

# ================= 配置参数 =================
root_dir = r"E:\YOLO_DataSet\yu-new"  # 原始数据根目录
default_num_aug_per_image = 3          # 默认每张图增强次数
target_total_images = 500              # 可选参数：总图像目标数量 (如果设置，将覆盖 default_num_aug_per_image)
max_aug_per_image_limit = 10           # 新增：单张图片增强的最大次数限制
max_retry_per_image = 5                # 每张图每次增强最大重试次数
min_bbox_visibility = 0.1              # 边界框最小可见比例
output_folder_name = "Albumentations_plus"  # 输出增强目录

# ================= Albumentations 增强 pipeline =================
# 注意：RandomLightCircle 类的 __init__ 方法可能需要调整以适应 Albumentations 的 transform API
# 例如，如果它直接修改图像，可以将其封装在一个自定义的 A.ImageOnlyTransform 中
# 为了简化，这里假设 RandomLightCircle 兼容 Albumentations 的自定义 transform 模式
albumentations_transform = A.Compose([
    # --- 几何变换 ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Affine(scale=(0.9, 1.1),
             translate_percent=(0.05, 0.05),
             rotate=(-15, 15),
             shear=(-3, 3), p=0.3),

    # --- 颜色与光照 ---
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=(-30, 0), g_shift_limit=(-10, 10), b_shift_limit=(0, 30), p=0.2),

    # --- 自定义光斑 ---
    AdvancedRandomLightCircle(
        max_radius_ratio=0.4,  # 光斑最大半径占比
        intensity_range=(0.6, 1.4),  # 光斑强度范围
        color_choices=[(255, 255, 255),  # 白光
                       (255, 240, 200),  # 暖黄
                       (200, 220, 255)],  # 冷白
        blur_limit=(21, 51),  # 高斯模糊核大小范围
        num_spots=(1, 3),  # 每张图随机生成 1~3 个光斑
        scatter=True,  # 是否启用中心亮、边缘渐暗效果
        blue_green_shift=True,  # 是否模拟水下红光吸收
        p=0.4  # 增强概率
    ),

    # --- 模糊 & 噪声 ---
    A.MotionBlur(blur_limit=3, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    # GaussNoise 的 std_range 应为 [0,1] 范围内的浮点数，Albumentations 会自动乘以 255
    A.GaussNoise(std_range=(3/255, 15/255), mean_range=(-0.01, 0.01), per_channel=True, p=0.4), # var_limit 对应方差，近似 std_range=(sqrt(0.001), sqrt(0.015))
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),

    # --- 水下畸变 ---
    A.ElasticTransform(alpha=30, sigma=5, p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
    A.OpticalDistortion(distort_limit=0.15, p=0.2),
    A.Affine(translate_percent=(0.02, 0.02), scale=(1.0, 1.0), rotate=0, shear=0, p=0.15), # 小平移仿射

    # --- 清晰度增强 ---
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.2), p=0.3),
],
    p=1.0,
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=min_bbox_visibility)
)

# ================= 自定义旋转函数（扩展画布） =================
def rotate_expand(img, bboxes, angle):
    """
    旋转图像并扩展画布以避免裁剪，同时更新YOLO格式的边界框。
    Args:
        img (np.array): 输入图像。
        bboxes (list): YOLO格式的边界框列表，每个元素为 [x_center, y_center, width, height]。
        angle (float): 旋转角度（度）。
    Returns:
        tuple: (rotated_img, new_bboxes) 旋转后的图像和更新后的边界框列表。
    """
    h, w = img.shape[:2]
    # 计算新画布的对角线长度，以确保能容纳旋转后的图像
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))
    canvas = np.zeros((diag, diag, 3), dtype=img.dtype)

    # 将原图像放置在新画布中心
    x_offset = (diag - w) // 2
    y_offset = (diag - h) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img

    center = (diag // 2, diag // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) # 旋转矩阵
    rotated_img = cv2.warpAffine(canvas, M, (diag, diag), borderValue=(0, 0, 0)) # 执行旋转

    new_bboxes = []
    for (x_c, y_c, bw, bh) in bboxes:
        # 将归一化坐标转换为绝对坐标
        abs_x = x_c * w + x_offset
        abs_y = y_c * h + y_offset
        abs_w = bw * w
        abs_h = bh * h

        # 计算边界框的四个角点
        box_pts = np.array([
            [abs_x - abs_w / 2, abs_y - abs_h / 2],
            [abs_x + abs_w / 2, abs_y - abs_h / 2],
            [abs_x + abs_w / 2, abs_y + abs_h / 2],
            [abs_x - abs_w / 2, abs_y + abs_h / 2]
        ])

        # 将角点应用旋转矩阵
        ones = np.ones((box_pts.shape[0], 1))
        pts_ones = np.hstack([box_pts, ones])
        rotated_pts = M.dot(pts_ones.T).T

        # 计算旋转后边界框的最小外接矩形
        x_min, y_min = rotated_pts[:, 0].min(), rotated_pts[:, 1].min()
        x_max, y_max = rotated_pts[:, 0].max(), rotated_pts[:, 1].max()

        # 转换为新的YOLO格式 (归一化到新画布尺寸)
        cx = (x_min + x_max) / 2 / diag
        cy = (y_min + y_max) / 2 / diag
        bw_new = (x_max - x_min) / diag
        bh_new = (y_max - y_min) / diag

        # 确保边界框在 [0,1] 范围内
        cx, cy, bw_new, bh_new = map(lambda v: min(max(v, 0.0), 1.0), [cx, cy, bw_new, bh_new])

        # 过滤掉无效或过小的边界框
        if bw_new > 0.001 and bh_new > 0.001: # 设定一个很小的阈值
            new_bboxes.append([cx, cy, bw_new, bh_new])

    return rotated_img, new_bboxes

# ================= 单张图像增强函数 =================
def process_image(img_file, labels_dir, output_img_dir, output_lbl_dir,
                  num_aug_per_image, global_stats, folder_stats, failure_list, lock):
    """
    处理单张图像的增强任务，包括读取、增强、保存和统计。
    Args:
        img_file (str): 原始图像文件路径。
        labels_dir (str): 原始标签目录路径。
        output_img_dir (str): 输出增强图像目录路径。
        output_lbl_dir (str): 输出增强标签目录路径。
        num_aug_per_image (int): 每张图的增强次数。
        global_stats (dict): 全局统计共享字典。
        folder_stats (dict): 文件夹级统计共享字典。
        failure_list (list): 失败日志共享列表。
        lock (Lock): 进程锁。
    Returns:
        None
    """
    base = os.path.splitext(os.path.basename(img_file))[0]
    label_file = os.path.join(labels_dir, base + ".txt")
    img = cv2.imread(img_file) # 使用 OpenCV 读取图像

    # 检查图像和标签文件是否存在且可读
    if img is None:
        with lock:
            failure_list.append((img_file, "无法读取图像"))
            folder_stats["skipped_images"] += 1
        return
    if not os.path.exists(label_file):
        with lock:
            failure_list.append((img_file, "标签文件缺失"))
            folder_stats["skipped_images"] += 1
        return

    # 读取原始边界框
    bboxes, class_labels = [], []
    try:
        with open(label_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError(f"标签格式错误: {line.strip()}")
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            # 确保边界框坐标在 [0,1] 范围内
            x, y, w, h = [min(max(v, 0.0), 1.0) for v in (x, y, w, h)]
            if w > 0 and h > 0: # 过滤掉无效尺寸的边界框
                bboxes.append([x, y, w, h])
                class_labels.append(cls)
            else:
                with lock:
                    failure_list.append((img_file, f"原始标签包含无效尺寸边界框: {line.strip()}"))
    except Exception as e:
        with lock:
            failure_list.append((img_file, f"读取标签或格式错误: {e}"))
            folder_stats["skipped_images"] += 1
        return

    if not bboxes: # 原始图像没有有效边界框
        with lock:
            failure_list.append((img_file, "原始图像无有效边界框，跳过增强"))
            folder_stats["skipped_images"] += 1
        return

    with lock:
        folder_stats["original_images"] += 1

    for aug_idx in range(num_aug_per_image):
        success = False
        current_aug_discarded_bboxes = 0 # 记录本次增强丢弃的边界框数量

        with lock:
            folder_stats["actual_aug_attempts"] += 1

        for attempt in range(max_retry_per_image):
            temp_img, temp_bboxes, temp_labels = img.copy(), bboxes[:], class_labels[:]

            # 随机旋转 (p=0.75)
            if random.random() < 0.75:
                angle = random.uniform(-30, 30)
                temp_img, temp_bboxes = rotate_expand(temp_img, temp_bboxes, angle)
                # 旋转后如果边界框全部消失，则本次尝试失败
                if not temp_bboxes:
                    continue

            try:
                # 执行 Albumentations 增强
                transformed = albumentations_transform(image=temp_img,
                                                       bboxes=temp_bboxes,
                                                       class_labels=temp_labels)
                aug_img = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_labels = transformed["class_labels"]

                if not aug_bboxes: # 增强后无有效边界框
                    raise ValueError("Albumentations 增强后无有效边界框")

                current_aug_discarded_bboxes = len(bboxes) - len(aug_bboxes)
                success = True
                break # 增强成功，跳出重试循环

            except Exception as e:
                # 记录详细的失败信息，但继续重试
                if attempt == max_retry_per_image - 1: # 最后一次尝试失败才记录
                     with lock:
                        failure_list.append((img_file, f"增强失败 (第{aug_idx+1}次, 尝试{attempt+1}/{max_retry_per_image}): {e}"))
                continue

        if success:
            # 保存增强图像和标签
            save_img_path = os.path.join(output_img_dir, f"{base}_aug{aug_idx + 1}.jpg")
            save_lbl_path = os.path.join(output_lbl_dir, f"{base}_aug{aug_idx + 1}.txt")
            cv2.imwrite(save_img_path, aug_img)
            with open(save_lbl_path, "w") as f:
                for cls, (x, y, w, h) in zip(aug_labels, aug_bboxes):
                    f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            with lock:
                global_stats["total_success_augs"] += 1
                global_stats["total_discarded_bboxes"] += current_aug_discarded_bboxes
                folder_stats["successful_augs"] += 1
                folder_stats["discarded_bboxes_count"] += current_aug_discarded_bboxes
        else:
            with lock:
                global_stats["total_failure_augs"] += 1
                folder_stats["failed_augs"] += 1

# ================= 遍历文件夹函数 =================
def get_coral_folders(root_dir):
    """
    遍历给定根目录，寻找包含 'images' 和 'labels' 子文件夹的数据集目录。
    支持两种模式：root_dir 本身就是数据集目录，或者 root_dir 下的子目录是数据集目录。
    """
    folders = []
    # 检查 root_dir 本身是否是数据集目录
    if os.path.isdir(os.path.join(root_dir, "images")) and os.path.isdir(os.path.join(root_dir, "labels")):
        folders.append(root_dir)
    # 遍历子目录
    for d in os.listdir(root_dir):
        full_path = os.path.join(root_dir, d)
        if os.path.isdir(full_path) and \
           os.path.isdir(os.path.join(full_path, "images")) and \
           os.path.isdir(os.path.join(full_path, "labels")):
            folders.append(full_path)
    return folders

# ================= 主程序 =================
if __name__ == "__main__":
    freeze_support()  # Windows 多进程必须

    manager = Manager()
    # 全局统计
    global_stats = manager.dict({
        "total_original_images": 0,
        "total_aug_attempts": 0,       # 实际执行的增强次数 (num_aug_per_image * 原始图像数)
        "total_success_augs": 0,       # 成功生成增强图像的次数
        "total_failure_augs": 0,       # 彻底失败（重试后仍失败）的增强次数
        "total_discarded_bboxes": 0,   # 所有成功增强中丢弃的边界框总数
        "total_original_bboxes_in_success_augs": 0 # 用于计算总体的边界框丢弃率
    })
    failure_list = manager.list() # 记录失败详情
    lock = manager.Lock() # 用于保护共享变量的锁

    cpu_cores = max(1, cpu_count() // 2) # 通常保留一个核心给系统或其他进程
    print(f"⚡ CPU 核心数: {cpu_count()}, 将使用 max_workers={cpu_cores}")

    all_coral_folders = get_coral_folders(root_dir)
    if not all_coral_folders:
        print("❌ 未在指定根目录或其子目录中找到包含 'images' 和 'labels' 的数据集文件夹。请检查 'root_dir' 配置。")
        exit(0)

    for folder_idx, folder in enumerate(all_coral_folders):
        print(f"\n--- 📁 处理文件夹 [{folder_idx+1}/{len(all_coral_folders)}]: {folder} ---")
        input_img_dir = os.path.join(folder, "images")
        input_lbl_dir = os.path.join(folder, "labels")

        img_files = [os.path.join(input_img_dir, f)
                     for f in os.listdir(input_img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

        if len(img_files) == 0:
            print("⚠️ 该文件夹没有找到支持的图像文件，跳过。")
            continue

        # 初始化文件夹级统计
        folder_stats = manager.dict({
            "original_images": 0, # 实际处理的原始图像数量（有有效标签的）
            "skipped_images": 0,  # 因文件缺失或标签无效而跳过的原始图像
            "actual_aug_attempts": 0, # 对当前文件夹图像实际进行的增强尝试总数
            "successful_augs": 0,     # 当前文件夹中成功生成增强图像的次数
            "failed_augs": 0,         # 当前文件夹中彻底失败的增强次数
            "discarded_bboxes_count": 0, # 当前文件夹中成功增强时丢弃的边界框总数
        })

        # 动态计算增强次数 (修改后的逻辑)
        current_num_aug_per_image = default_num_aug_per_image  # 默认值
        if target_total_images is not None and target_total_images > 0 and len(img_files) > 0:
            # 计算需要额外生成的增强图片数量
            num_additional_aug_needed = target_total_images - len(img_files)
            if num_additional_aug_needed < 0:
                # 如果目标总数小于原始图像数，则不进行增强 (或只进行一次，取决于需求)
                current_num_aug_per_image = 0
                print(f"⚠️ 目标总图像数 ({target_total_images}) 小于原始图像数 ({len(img_files)})，跳过增强。")
            else:
                # 平均分配所需增强次数到每张原始图像
                current_num_aug_per_image = max(0, math.ceil(num_additional_aug_needed / len(img_files)))
        elif len(img_files) == 0:
            current_num_aug_per_image = 0  # 没有原始图像，无需增强

        # 应用单张图片最大增强次数限制
        if max_aug_per_image_limit is not None and max_aug_per_image_limit > 0:
            current_num_aug_per_image = min(current_num_aug_per_image, max_aug_per_image_limit)

        print(f"📦 原始图像数量: {len(img_files)}")
        print(f"✨ 每张图增强次数 (动态计算/默认, 已应用最大限制): {current_num_aug_per_image}")
        if current_num_aug_per_image <= 0:
            print(f"✅ 增强次数为0，跳过该文件夹的增强。")
            continue

        output_img_dir = os.path.join(folder, output_folder_name, "images")
        output_lbl_dir = os.path.join(folder, output_folder_name, "labels")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)

        # 记录原始图像总数
        with lock:
            global_stats["total_original_images"] += len(img_files)

        # 多进程池执行
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            list(tqdm(executor.map(
                process_image,
                img_files,
                [input_lbl_dir] * len(img_files),
                [output_img_dir] * len(img_files),
                [output_lbl_dir] * len(img_files),
                [current_num_aug_per_image] * len(img_files),
                [global_stats] * len(img_files),
                [folder_stats] * len(img_files),
                [failure_list] * len(img_files),
                [lock] * len(img_files),
            ), total=len(img_files), desc=f"Augmenting {os.path.basename(folder)}", ncols=100))

        # 文件夹级统计报告
        print("\n📊 文件夹级增强统计:")
        print(f"  原始图像总数 (计划处理): {len(img_files)}")
        print(f"  实际处理原始图像数 (有有效标签): {folder_stats['original_images']}")
        print(f"  跳过原始图像数 (文件或标签问题): {folder_stats['skipped_images']}")
        print(f"  增强尝试次数 (共 {folder_stats['original_images']} 张图, 每图 {current_num_aug_per_image} 次): {folder_stats['actual_aug_attempts']}")
        print(f"  成功增强图像数: {folder_stats['successful_augs']}")
        print(f"  彻底失败增强数: {folder_stats['failed_augs']}")
        print(f"  成功率: {folder_stats['successful_augs'] / folder_stats['actual_aug_attempts'] * 100:.2f}%" if folder_stats['actual_aug_attempts'] > 0 else "0.00%")
        print(f"  增强中丢弃边界框总数: {folder_stats['discarded_bboxes_count']}")
        if folder_stats['successful_augs'] > 0:
            # 假设平均每张原始图像有 approx_bboxes 个边界框
            # 粗略计算成功增强中原始边界框的总数，用于计算丢弃率
            # 这是一个近似值，因为我们没有精确统计每张原始图的初始 bbox 数量
            # 更精确的统计需要在 process_image 中将初始 bbox 数量也传递到统计
            avg_original_bboxes_per_image = 0
            if folder_stats['original_images'] > 0:
                # 再次读取标签文件以获取每个文件的初始 bbox 数量
                initial_bboxes_count = 0
                for img_file in img_files:
                    base = os.path.splitext(os.path.basename(img_file))[0]
                    label_file = os.path.join(input_lbl_dir, base + ".txt")
                    if os.path.exists(label_file):
                        try:
                            with open(label_file, 'r') as f_lbl:
                                initial_bboxes_count += sum(1 for line in f_lbl if line.strip())
                        except Exception:
                            pass # 忽略读取标签的错误，因为主流程已经处理过
                if initial_bboxes_count > 0:
                    avg_original_bboxes_per_image = initial_bboxes_count / folder_stats['original_images']

            # 估算总的原始边界框数，用于计算丢弃率
            estimated_total_original_bboxes = int(avg_original_bboxes_per_image * folder_stats['successful_augs'])
            if estimated_total_original_bboxes > 0:
                bbox_discard_rate = folder_stats['discarded_bboxes_count'] / estimated_total_original_bboxes * 100
                print(f"  边界框丢弃率 (在成功增强中): {bbox_discard_rate:.2f}%")
                if bbox_discard_rate > 20: # 高丢弃率警告阈值
                    print("  ⚠️ 警告: 边界框丢弃率较高，可能需要调整 min_bbox_visibility 或增强参数。")
            else:
                print("  边界框丢弃率: 无法计算 (无原始边界框数据)")
        else:
            print("  边界框丢弃率: 无法计算 (无成功增强)")

    # ================= 全局统计报告 =================
    print("\n\n=============== 🚀 全局增强报告 ================")
    print(f"总原始图像数: {global_stats['total_original_images']}")
    print(f"总增强尝试次数: {global_stats['total_success_augs'] + global_stats['total_failure_augs']}")
    print(f"成功生成增强图像总数: {global_stats['total_success_augs']}")
    print(f"彻底失败的增强任务总数: {global_stats['total_failure_augs']}")
    total_effective_augs = global_stats['total_success_augs'] + global_stats['total_failure_augs']
    success_rate = global_stats['total_success_augs'] / total_effective_augs * 100 if total_effective_augs > 0 else 0
    print(f"总体增强成功率: {success_rate:.2f}%")
    print(f"所有成功增强中丢弃的边界框总数: {global_stats['total_discarded_bboxes']}")
    # 这里的原始边界框总数统计比较复杂，需要遍历所有原始标签。
    # 简单起见，可以近似地将每个成功增强的图像视为携带了原始数量的边界框。
    # 更精确的计算需要修改 process_image 来返回每个原始图像的初始bbox数量。
    print(" (注意：此处边界框丢弃率计算基于成功增强过程中的内部统计)")


    # ================= 失败日志 =================
    if len(failure_list) > 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"augmentation_failure_log_{timestamp}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            for img_path, reason in failure_list:
                f.write(f"{img_path}\t{reason}\n")
        print(f"\n⚠️ 部分图像增强失败，详情请查看日志文件: {log_file}")
    else:
        print("\n🎉 所有图像增强任务均成功完成！")

    print("\n程序运行完毕。")