import torch

def generate_gt_masks(instance_map, ignore_id=0):
    """
    Args:
        instance_map: [bs, H, W] 实例ID图
        ignore_id: 需要忽略的背景ID

    Returns:
        gt_masks: [bs, N, H, W] 二进制mask张量
        valid_counts: [bs] 每个样本的有效实例数
    """
    bs, H, W = instance_map.shape
    device = instance_map.device

    # 存储每个batch的mask列表
    batch_masks = []
    valid_counts = []

    for b in range(bs):
        # 获取当前样本的实例ID图
        instance_ids = instance_map[b].unique()

        # 过滤背景ID
        valid_ids = instance_ids[instance_ids != ignore_id]
        num_instances = len(valid_ids)
        valid_counts.append(num_instances)

        # 生成二进制mask
        masks = []
        for inst_id in valid_ids:
            mask = (instance_map[b] == inst_id)  # [H, W]
            masks.append(mask)

        # 转换为张量 [N, H, W]
        if len(masks) > 0:
            masks_tensor = torch.stack(masks, dim=0)  # [N, H, W]
        else:
            masks_tensor = torch.zeros((0, H, W), device=device)

        batch_masks.append(masks_tensor)

    # 找到最大实例数
    # max_instances = max(valid_counts) if len(valid_counts) > 0 else 0
    max_instances = 95

    # 填充为统一尺寸
    padded_masks = []
    for masks in batch_masks:
        if masks.shape[0] < max_instances:
            padding = torch.zeros(
                (max_instances - masks.shape[0], H, W),
                dtype=torch.bool,
                device=device
            )
            padded = torch.cat([masks, padding], dim=0)
        else:
            padded = masks
        padded_masks.append(padded)

    # 合并为bs维度 [bs, N, H, W]
    gt_masks = torch.stack(padded_masks, dim=0)

    return gt_masks, torch.tensor(valid_counts, device=device)

if __name__ == "__main__":
    # 假设输入实例图:bs=2, N=100, h=2, w=3
    instance_map = torch.tensor([
        [[0, 1, 1],  # 样本1
        [0, 2, 2]],

        [[3, 3, 0],  # 样本2
        [4, 4, 5]]
    ], dtype=torch.long)

    gt_masks, counts = generate_gt_masks(instance_map)

    print("GT Masks shape:", gt_masks.shape)  # [2, 3, 2, 3]
    print("Valid counts:", counts)           # [2, 3]

    # 可视化第一个样本的mask
    sample1_masks = gt_masks[0, :counts[0]]
    print("Sample1 masks:\n", sample1_masks.int().squeeze())
    # 输出：
    # Sample1 masks:
    # tensor([[[0, 1, 1],
    #      [0, 0, 0]],

    #     [[0, 0, 0],
    #      [0, 1, 1]]], dtype=torch.int32)
