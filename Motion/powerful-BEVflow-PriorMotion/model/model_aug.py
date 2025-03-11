# import torch
# import torch.nn.functional as F

# # 假设的BEV特征图和速度场预测
# bev_feature = torch.randn(16, 32, 256, 256)  # (batch_size, channels, height, width)
# disp_pred = torch.randn(16, 20, 2, 256, 256)  # (batch_size, time_steps, 2, height, width)
# disp_gt = torch.randn(16, 20, 2, 256, 256)  # (batch_size, time_steps, 2, height, width)

# # 定义旋转角度
# angles = [90, 180, 270]

# # 旋转函数
# def rotate_tensor(tensor, angle):
#     if angle == 90:
#         return tensor.permute(0, 1, 2, 4, 3).flip(3)
#     elif angle == 180:
#         return tensor.flip(3).flip(4)
#     elif angle == 270:
#         return tensor.permute(0, 1, 2, 4, 3).flip(4)
#     else:
#         return tensor

# # 逆旋转函数
# def inverse_rotate_tensor(tensor, angle):
#     if angle == 90:
#         return tensor.permute(0, 1, 2, 4, 3).flip(4)
#     elif angle == 180:
#         return tensor.flip(4).flip(3)
#     elif angle == 270:
#         return tensor.permute(0, 1, 2, 4, 3).flip(3)
#     else:
#         return tensor

# # 模型前向传播函数（假设模型已经定义好）
# def model_forward(bev_feature):
#     # 这里假设模型直接返回速度场预测
#     return torch.randn_like(disp_pred)

# # 计算方向一致性损失
# def directional_consistency_loss(bev_feature, disp_pred, disp_gt, angles):
#     batch_size, T, _, H, W = disp_pred.shape
#     total_loss = 0
#     for angle in angles:
#         # 旋转BEV特征图
#         rotated_bev_feature = rotate_tensor(bev_feature.unsqueeze(1), angle).squeeze(1)
#         # 通过模型生成旋转后的预测速度场
#         rotated_disp_pred = model_forward(rotated_bev_feature)
#         # 逆旋转预测速度场
#         inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
#         # 确保形状匹配
#         assert inv_rotated_disp_pred.shape == disp_gt.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"
#         # 计算L1损失
#         loss = F.l1_loss(inv_rotated_disp_pred, disp_gt)
#         total_loss += loss
#     return total_loss / len(angles)

# # 计算方向一致性损失
# loss = directional_consistency_loss(bev_feature, disp_pred, disp_gt, angles)
# print(f"Directional Consistency Loss: {loss.item()}")


import torch
import torch.nn.functional as F

# 假设的BEV特征图和速度场预测
bev_feature = torch.randn(16, 20, 128, 64, 64)  # (batch_size, time_steps, channels, height, width)
disp_pred = torch.randn(16, 20, 2, 64, 64)  # (batch_size, time_steps, 2, height, width)
disp_gt = torch.randn(16, 20, 2, 64, 64)  # (batch_size, time_steps, 2, height, width)

# 定义旋转角度
angles = [90, 180, 270]

# 旋转函数
def rotate_tensor(tensor, angle):
    if angle == 90:
        return tensor.permute(0, 1, 2, 4, 3).flip(4)
    elif angle == 180:
        return tensor.flip(3).flip(4)
    elif angle == 270:
        return tensor.permute(0, 1, 2, 4, 3).flip(3)
    else:
        return tensor

# 逆旋转函数
def inverse_rotate_tensor(tensor, angle):
    if angle == 90:
        return tensor.permute(0, 1, 2, 4, 3).flip(3)
    elif angle == 180:
        return tensor.flip(4).flip(3)
    elif angle == 270:
        return tensor.permute(0, 1, 2, 4, 3).flip(4)
    else:
        return tensor

# 模型前向传播函数（假设模型已经定义好）
def model_forward(bev_feature):
    # 这里假设模型直接返回速度场预测
    return torch.randn_like(disp_pred)

# 计算方向一致性损失
def directional_consistency_loss(bev_feature, disp_pred, disp_gt, angles):
    batch_size, T, _, H, W = disp_pred.shape
    total_loss = 0
    for angle in angles:
        # 旋转BEV特征图
        rotated_bev_feature = rotate_tensor(bev_feature, angle)
        # 通过模型生成旋转后的预测速度场
        rotated_disp_pred = model_forward(rotated_bev_feature)
        # 逆旋转预测速度场
        inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
        # 确保形状匹配
        assert inv_rotated_disp_pred.shape == disp_gt.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"
        # 计算L1损失
        loss = F.l1_loss(inv_rotated_disp_pred, disp_gt)
        total_loss += loss
        # print(inv_rotated_disp_pred.shape)
    return total_loss / len(angles)

# 计算方向一致性损失
# loss = directional_consistency_loss(bev_feature, disp_pred, disp_gt, angles)
# print(f"Directional Consistency Loss: {loss.item()}")