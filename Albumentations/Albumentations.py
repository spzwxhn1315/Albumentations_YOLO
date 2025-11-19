import os
import cv2
import albumentations as A
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ========= 配置部分 =========
input_dir = r"E:\YOLO_DataSet_Augmentation\coral_soft\images"  # 输入文件夹
output_dir = r"E:\YOLO_DataSet_Augmentation\coral_soft\plus\images"  # 输出文件夹
num_aug_per_image = 3  # 每张图片生成多少个增强版本
num_workers = 6  # 并行进程数，可改成 CPU 核心数

# ========= 数据增强管线 =========
transform = A.Compose([
    # --- 几何变换 ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(0.1, 0.1),
        rotate=(-30, 30),
        shear=(-10, 10),
        cval=0,  # 控制图像边界填充颜色
        mask_interpolation=0,  # mask 用最近邻填充
        fit_output=False,
        p=0.7
    ),


    # --- 颜色增强 ---
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),

    # --- 水下环境模拟 ---
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.ISONoise(color_shift=(0.01, 0.05), p=0.3),
    A.HueSaturationValue(hue_shift_limit=20,
                         sat_shift_limit=30,
                         val_shift_limit=20, p=0.5)
], p=1.0)


# ========= 单张图片处理函数 =========
def process_image(img_name):
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)

    if image is None:
        return f"跳过无法读取的文件: {img_path}"

    # 保存原图（可选）
    # cv2.imwrite(os.path.join(output_dir, img_name), image)

    # 生成增强版本
    for i in range(num_aug_per_image):
        augmented = transform(image=image)["image"]
        save_name = f"{os.path.splitext(img_name)[0]}_aug{i + 1}.jpg"
        cv2.imwrite(os.path.join(output_dir, save_name), augmented)

    return f"完成 {img_name}"


# ========= 主流程 =========
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_image, img_files), total=len(img_files)))

    print("✅ 数据增强完成！")
