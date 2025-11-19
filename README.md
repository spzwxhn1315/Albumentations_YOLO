# Albumentations YOLO Augmentation Plus

这是一个为 **YOLO 检测任务** 专门设计的 **高强度、多策略数据增强工具**。
支持：

* Albumentations 全套增强
* 自定义旋转 + 自动扩展画布（避免 bbox 裁切）
* YOLO 格式 bbox 自动适配与裁剪
* 多进程高速并行增强（CPU 自动识别）
* 动态增强策略（根据目标图片总数自动计算增强次数）

适用于：
水下图像、珊瑚检测、复杂键目标检测、类不均衡补充、数据扩增等场景。

---

## ✨ 功能特点

### 1. **全套 Albumentations 增强**

包括翻转、亮度/对比度、颜色扰动、模糊、运动模糊、噪声、弹性形变、网格形变、仿射、遮挡等数十种操作。

### 2. **自定义旋转（可扩展画布）**

避免 YOLO bbox 因旋转被裁剪的问题。

### 3. **增强失败重试机制**

若增强后 bbox 不可见，会自动重试，最多尝试 `max_retry_per_image` 次。

### 4. **多进程加速**

自动检测 CPU 核心，默认保留 2 个核心给系统。

### 5. **YOLO 目录结构自动识别**

可直接输入一个父目录，程序将自动找到内部包含 `images/labels` 的珊瑚文件夹。

### 6. **增强次数自动计算**

根据：

```python
target_total_images = 500
```

和原始图片数量自动计算每张图的增强次数。

---

## 🗂️ 项目目录结构

你的数据集结构应如下：

```
coral-new/
    coral01/
        images/
        labels/
    coral02/
        images/
        labels/
    ...
```

增强输出会自动生成：

```
coral-new/
    coral01/
        Albumentations_plus/
            images/
            labels/
```

---

## 🚀 使用方式

### 1. 克隆项目

```bash
git clone https://github.com/spzwxhn1315/Albumentations_YOLO.git
cd Albumentations_YOLO
```

### 2. 安装环境

```bash
pip install -r requirements.txt
```

### 3. 运行增强脚本

```bash
python augment_yolo.py
```

### 4. 脚本参数说明

编辑脚本顶部参数：

```python
root_dir = r"E:\YOLO_DataSet\coral-new"  # 数据集根目录
default_num_aug_per_image = 3           # 每张图默认增强次数
target_total_images = 500               # 想生成多少张？自动计算增强次数
max_retry_per_image = 5                 # bbox 丢失重试次数
```

---

## 📊 全局统计输出示例

```
📊 全局统计: 丢弃 12 / 1800 bboxes (0.67%)
```

帮助你评估增强质量和 bbox 损失情况。

---

## 📢 注意事项

* 本工具只支持 **YOLO txt 格式**（class x y w h）
* 如果增强后 bbox 全部消失，会自动重试增强
* 如使用 **水下图像增强 / 珊瑚检测**，增强策略已经针对水下图像进行了优化

---

## 🧑‍💻 作者

如果你愿意，我可以帮你补充作者说明、项目介绍、Badge、Star CTA、Logo、示例数据说明。

---

## ⭐ 如果项目对你有帮助

欢迎 Star 支持！

---


