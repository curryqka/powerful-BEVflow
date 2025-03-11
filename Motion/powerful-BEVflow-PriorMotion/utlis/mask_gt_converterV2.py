import numpy as np
import torch

def generate_gt_masks(instance_map, ignore_id=0, max_instances=100, use_batch_dim=True, non_empty_maps=None):
    """
    Args:
        instance_map: [H, W] 或者 [bs, H, W] 实例ID图(取决于use_batch_dim)
        ignore_id: 需要忽略的背景ID
        use_batch_dim: 是否考虑batch size这个维度

    Returns:
        gt_masks: 如果use_batch_dim为True,则返回[bs, N, H, W]；否则返回[N, H, W]
        valid_counts: 每个样本的有效实例数(如果use_batch_dim为False,则返回单个整数)
    """
    if use_batch_dim:
        bs, H, W = instance_map.shape
        device = 'cpu'  # 假设数据在CPU上操作
        instance_map_tensor = torch.tensor(instance_map)
    else:
        H, W = instance_map.shape
        bs = 1
        device = 'cpu'
        instance_map_tensor = torch.tensor(instance_map).unsqueeze(0)  # 添加batch维度以便统一处理

    # 存储每个batch的mask列表
    batch_masks = []
    valid_counts = []

    instance_map_tensor = instance_map_tensor + 1
    for b in range(bs):
        # 获取当前样本的实例ID图
        instance_ids = instance_map_tensor[b].unique()

        # 过滤背景ID
        valid_ids = instance_ids[instance_ids != ignore_id]
        num_instances = len(valid_ids)
        valid_counts.append(num_instances)

        # 生成二进制mask
        masks = []
        for inst_id in valid_ids:
            mask = (instance_map_tensor[b] == inst_id)  # [H, W]
            if non_empty_maps is not None:
                mask = mask * non_empty_maps[b]
            masks.append(mask)

        # 转换为张量 [N, H, W]
        if len(masks) > 0:
            masks_tensor = torch.stack(masks, dim=0)  # [N, H, W]
        else:
            masks_tensor = torch.zeros((0, H, W), dtype=torch.bool)

        batch_masks.append(masks_tensor)

    # 找到最大实例数
    # max_instances = 100  # 设置一个固定的最大实例数，避免动态计算

    # 填充为统一尺寸
    padded_masks = []
    for masks in batch_masks:
        if masks.shape[0] < max_instances:
            padding = torch.zeros(
                (max_instances - masks.shape[0], H, W),
                dtype=torch.bool
            )
            padded = torch.cat([masks, padding], dim=0)
        else:
            padded = masks
        padded_masks.append(padded.numpy())  # 转换回numpy结构

    if use_batch_dim:
        # 合并为bs维度 [bs, N, H, W]
        gt_masks = np.stack(padded_masks, axis=0)
        return gt_masks, np.array(valid_counts)
    else:
        # 移除额外添加的batch维度并返回结果
        return padded_masks[0], valid_counts[0]

if __name__ == "__main__":
    # 示例输入：不使用batch维度
    instance_map_np = np.array([
        [0, 1, 1],
        [0, 2, 2]
    ], dtype=np.int64)

    gt_masks, count = generate_gt_masks(instance_map_np, use_batch_dim=False)

    print("GT Masks shape:", gt_masks.shape)  # [N, H, W]
    print("Valid counts:", count)  # 单个整数

    # 示例输入：使用batch维度
    instance_map_np_batch = np.array([
        [[0, 1, 1],  # 样本1
         [0, 2, 2]],

        [[3, 3, 0],  # 样本2
         [4, 4, 5]]
    ], dtype=np.int64)

    gt_masks_batch, counts_batch = generate_gt_masks(instance_map_np_batch, use_batch_dim=True)

    print("GT Masks shape with batch:", gt_masks_batch.shape)  # [bs, N, H, W]
    print("Valid counts with batch:", counts_batch)  # [bs]
