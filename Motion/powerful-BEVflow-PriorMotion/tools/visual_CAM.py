import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 假设 model 是你的神经网络模型，且最后一层是全连接层
# 并且你想要可视化的特征图来自于名为 'feature_extractor' 的模块

def visualize_cam(feature_map, weights, class_idx, img):
    """
    可视化类别激活映射(CAM)

    参数:
    - feature_map: [bs, c, h, w] 形状的特征图
    - weights: 全连接层的权重
    - class_idx: 要可视化的类别索引
    - img: 原始图像
    """
    bs, c, h, w = feature_map.shape
    cam = torch.zeros((h, w), dtype=torch.float)

    # 计算 CAM
    for i in range(c):
        cam += weights[class_idx, i] * feature_map[0, i, :, :]

    # 缩放 CAM 到原始图像大小
    cam = cam.detach().numpy()
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / cam.max()  # Normalize between 0-1

    # 将 CAM 转换为 RGB 格式
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)

# 示例数据
feature_map = torch.rand(1, 32, 256, 256)  # 随机生成特征图作为示例
weights = torch.rand(10, 32)  # 假设有 10 类，随机生成权重
class_idx = 1  # 示例类别索引
img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # 随机生成示例图像

cam_img = visualize_cam(feature_map, weights, class_idx, img)

# 绘制图像和热力图
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
ax[1].set_title('CAM Overlay')
# plt.show()
plt.savefig(os.path.join("/SHARE/qiankangan/bevflow/visualization", str(0) + ".png"))
