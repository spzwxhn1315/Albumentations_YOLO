# Albumentations YOLO Augmentation Plus

高性能 YOLO 数据增强工具（支持水下图像增强 + 多进程加速）

本项目是一个面向 **目标检测（YOLO 系列）** 的高扩展性数据增强工具，
结合 **Albumentations 强增强**, **自定义旋转扩展画布**, **水下光照模型**,
并使用 **多进程 ProcessPoolExcutor 高速并行增强**，旨在为真实场景中的复杂图像提供稳定、可控的增强方式。

适用于：

* 水下图像增强 / 珊瑚检测
* 类不均衡数据扩充
* 小样本增强
* 研究场景的复杂变换（旋转、光照、噪声、多重空间增强）

---

## 🌟 功能特点

### 1. **Albumentations 全套增强**

包括但不限于：

* Rotate / ShiftScaleRotate / HorizontalFlip / VerticalFlip
* CLAHE
* RGBShift, HueSaturationValue
* Motion Blur, Gaussian Blur
* RandomShadow, RandomSunFlare
* ElasticTransform, GridDistortion
* Cutout / GridDropout
* Perspective、仿射变换
* 色彩抖动、模糊、噪声等

增强全部支持 YOLO bbox 自动同步。

---

### 2. **自定义旋转（可扩展画布）

解决经典旋转导致 bbox 被裁切的问题**

* 自动计算旋转角度对应的新画布大小
* 校正旋转后 bbox 坐标
* 保证 bbox 不丢失（否则自动重试）

---

### 3. **水下图像光照增强（Underwater Light Transform）**

模拟水下：

* 漫散射
* 光照衰减
* 色偏
* 光移位

并自动同步标签，增强真实度。

---

### 4. **增强失败自动重试（max_retry_per_image）**

如果增强后的图像：

* 目标全部丢失
* bbox 越界
* 坐标无效

系统会自动重试最多 N 次，保证增强质量。

---

### 5. **多进程 ProcessPoolExecutor 超高速增强**

* 自动识别 CPU 核心
* 主进程管理全局统计量（失败计数 / bbox 丢失比例）
* 子进程并行处理每张图像
* 处理大型数据集时显著提速

适配 Windows / Linux / macOS。

---

### 6. **增强次数自动计算（基于目标总量）**

你可以指定：

```python
target_total_images = 500
```

程序会：

* 统计当前图片数量
* 自动计算每张图需要增强的次数
* 保证最终达到目标数量

无需手动调整增强策略。

---

## 📁 数据集结构（YOLO 格式）

原始数据：

```
root/
    coral01/
        images/
        labels/
    coral02/
        images/
        labels/
```

增强后自动生成：

```
root/
    coral01/
        Albumentations_plus/
            images/
            labels/
```

完全不影响原始数据。

---

## 🚀 使用方法

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/Albumentations_YOLO.git
cd Albumentations_YOLO
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 修改配置（脚本顶部）

```python
root_dir = r"E:\YOLO_DataSet\coral-new"  # 数据集目录
default_num_aug_per_image = 3
target_total_images = 500
max_retry_per_image = 5
```

### 4. 运行增强

```bash
python augment_yolo.py
```

---

## 📊 输出统计示例

```
📊 全局统计: 丢弃 12 / 1800 bboxes (0.67%)
📁 coral01: 增强 350 张
📁 coral02: 增强 420 张
```

用于评估增强质量。

---

## 🧪 适用研究方向

本项目特别适合用于：

* 水下图像增强
* 珊瑚识别 / 海洋生物检测
* 复杂场景下的目标检测鲁棒性增强
* 研究型模型训练（如物理先验模型、影像退化增强）

---

## 👨‍💻 作者信息

**MR. 叶（Ye）**
海南热带海洋学院
计算机科学与技术学院 · 研究生

📧 **[baojiangy@stu.hntou.edu.cn](mailto:baojiangy@stu.hntou.edu.cn)**

如有学术合作、项目交流，欢迎随时联系。

---

## ⭐ Star 支持

如果本工具对你有帮助，欢迎 Star 支持项目发展！

---

## 📄 许可证

你可根据需要选择：MIT / Apache 2.0 / GPL / CC0
需要我帮你选择最合适的许可证，也可以告诉我。

---

如果你想加入 **徽章（Badges）**, **项目 Logo**, **流程图**,
或生成 **中英双语 README**，告诉我即可。
