import cv2
import numpy as np

def dilated_tensor(raw, ):

    # 创建一个二值掩膜，其中类别为4或5的位置标记为1，其余位置为0
    binary_mask = np.where(np.logical_or(raw == 4, raw == 5), 1, 0).astype(np.uint8)

    # 定义结构元素，这里我们使用一个3x3的矩形作为示例
    kernel = np.ones((3, 3), np.uint8)

    size = (5, 5)
    # 7x7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 使用5x5,7x7的矩形核
    # 创建不同形状的结构元素
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)  # 矩形
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)  # 椭圆形
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)  # 十字形

    # 执行膨胀操作
    dilated_mask = cv2.dilate(binary_mask, ellipse_kernel, iterations=1)

    # 根据膨胀后的掩膜更新原始类别张量
    # 将原始类别张量中对应于膨胀后掩膜值为1的位置设置为5（或者你也可以选择设置为4，取决于具体需求）
    updated_category_tensor = np.copy(raw)
    updated_category_tensor[dilated_mask == 1] = 5


    return updated_category_tensor

if __name__ == "__main__":
    # 假设 category_tensor 是你的类别张量，形状为 [256, 256]
    category_tensor = np.random.randint(0, 10, (256, 256))  # 示例随机生成的数据
    output = dilated_tensor(category_tensor)
    # breakpoint()
    # 输出更新后的类别张量以供检查
    print(output)
