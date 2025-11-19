import cv2
import numpy as np
import random
from albumentations import DualTransform

class AdvancedRandomLightCircle(DualTransform):
    """
    水下光斑增强：
    - 多光斑叠加
    - 椭圆/不规则形状
    - 光散射模拟（中心亮，边缘渐暗）
    - 蓝绿偏移模拟水下吸收
    """
    def __init__(
        self,
        max_radius_ratio=0.4,
        intensity_range=(0.6, 1.4),
        color_choices=[(255, 255, 255), (255, 240, 200), (200, 220, 255)],
        blur_limit=(21, 51),
        num_spots=(1, 3),
        scatter=True,
        blue_green_shift=True,
        p=0.5,
        always_apply=False
    ):
        super().__init__(always_apply, p)
        self.max_radius_ratio = max_radius_ratio
        self.intensity_range = intensity_range
        self.color_choices = color_choices
        self.blur_limit = blur_limit
        self.num_spots = num_spots
        self.scatter = scatter
        self.blue_green_shift = blue_green_shift

    def apply(self, img, **params):
        h, w = img.shape[:2]
        img_float = img.astype(np.float32)

        num_spots = random.randint(self.num_spots[0], self.num_spots[1])
        for _ in range(num_spots):
            # 光斑中心
            center_x = random.uniform(0, w)
            center_y = random.uniform(0, h)

            # 光斑半径
            max_r = min(h, w) * self.max_radius_ratio
            radius_x = random.uniform(max_r*0.3, max_r)
            radius_y = radius_x * random.uniform(0.7, 1.3)  # 椭圆拉伸

            # 光斑颜色和强度
            light_color = np.array(random.choice(self.color_choices), dtype=np.float32)
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])

            # 创建 mask
            mask = np.zeros((h, w, 3), dtype=np.float32)

            # 使用椭圆绘制光斑
            cv2.ellipse(mask,
                        (int(center_x), int(center_y)),
                        (int(radius_x), int(radius_y)),
                        angle=random.uniform(0, 360),
                        startAngle=0, endAngle=360,
                        color=light_color,
                        thickness=-1)

            # 高斯模糊
            kernel_size = random.randrange(self.blur_limit[0], self.blur_limit[1]+1, 2)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

            # 中心亮，边缘渐暗
            if self.scatter:
                mask = mask / (mask.max() + 1e-6) * intensity
                mask = np.power(mask / 255.0, 1.2) * 255.0  # 指数衰减

            # 蓝绿偏移模拟水下
            if self.blue_green_shift:
                mask[..., 2] *= 0.6  # 红色衰减
                mask[..., 0] *= 1.0  # 蓝色略增强
                mask[..., 1] *= 1.0  # 绿色略增强

            # 叠加到原图
            img_float = np.clip(img_float * (1 - mask / 255.0) + mask, 0, 255)

        return img_float.astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("max_radius_ratio", "intensity_range", "color_choices",
                "blur_limit", "num_spots", "scatter", "blue_green_shift")
