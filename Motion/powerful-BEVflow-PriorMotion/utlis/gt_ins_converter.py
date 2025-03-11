import numpy as np
import torch

def _process_inputs(input_tensor, use_batch_dim, tensor_name):
    """统一处理输入张量的维度"""
    if isinstance(input_tensor, np.ndarray):
        tensor = torch.from_numpy(input_tensor)
    else:
        tensor = input_tensor.clone()

    # 自动添加batch维度
    if not use_batch_dim:
        if tensor.dim() == 3 and tensor_name == 'gt_classes':  # [C, H, W]
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2 and tensor_name == 'instance_maps':  # [H, W]
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2 and tensor_name == 'non_empty_maps':  # [H, W]
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 4 and tensor_name == 'displacements':  # [seq, H, W, 2]
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 4 and tensor_name == 'valid_map':  # [seq, H, W, 2]
            tensor = tensor.unsqueeze(0)
    return tensor

def convert_state_to_instance(gt_classes, instance_maps, max_instances, use_batch_dim=True, ignore_id=0, \
                              non_empty_maps=None):
    """
    改进后的状态转换函数
    返回：
        gt_ins: numpy数组或torch张量 [bs, max_instances, C]
        valid_counts: 有效实例数数组
    """
    # 输入处理
    gt_classes = _process_inputs(gt_classes, use_batch_dim, 'gt_classes')
    gt_classes = gt_classes.permute(0, 3, 1, 2)
    instance_maps = _process_inputs(instance_maps, use_batch_dim, 'instance_maps')
    non_empty_maps = _process_inputs(non_empty_maps, use_batch_dim, 'non_empty_maps')
    return_numpy = isinstance(gt_classes, np.ndarray)

    instance_maps = instance_maps + 1
    # print(instance_maps)

    bs, num_cls, h, w = gt_classes.shape
    device = gt_classes.device if hasattr(gt_classes, 'device') else 'cpu'

    # 收集有效实例
    valid_counts = []
    all_masks = []
    for b in range(bs):
        # print(instance_maps[b])
        instance_ids = torch.unique(instance_maps[b])
        valid_ids = instance_ids[instance_ids != ignore_id]
        valid_counts.append(len(valid_ids))
        # print(len(valid_ids))
        # 生成实例mask
        batch_masks = []
        for ins_id in valid_ids:
            mask = (instance_maps[b] == ins_id) * non_empty_maps[b]
            batch_masks.append(mask)
        all_masks.append(batch_masks)

    # 初始化输出张量
    gt_ins = torch.zeros((bs, max_instances, num_cls), device=device)

    # 填充数据
    for b in range(bs):
        for i, mask in enumerate(all_masks[b][:max_instances]):  # 截断超过max_instances的部分

            # breakpoint()
            masked = gt_classes[b] * mask.unsqueeze(0).float() * non_empty_maps[b]
            # masked = masked.
            # breakpoint()
            sum_cls = torch.sum(masked, dim=(1,2))
            # TODO:
            pixel_count = (mask * non_empty_maps[b]).sum()
            gt_ins[b, i] = sum_cls / pixel_count if pixel_count > 0 else 0
            # one_hot_vecoter = torch.zeros(( gt_classes[b].shape[0], ))
            # # breakpoint()
            # avg_vector = sum_cls / pixel_count if pixel_count > 0 else 0
            # cls_index = torch.argmax(avg_vector, dim=0)
            # one_hot_vecoter[cls_index] = 1
            # gt_ins[b, i] = one_hot_vecoter


            # 将张量展平成一维以便统计
            # masked = torch.randint(0, 10, (2, 100, 100))  # 示例数据
            # flat_masked = masked.view(-1)

            # # 找出唯一值及其对应的频率
            # unique_values, counts = flat_masked.unique(return_counts=True)

            # # 找到出现次数最多的值的索引
            # max_count_idx = torch.argmax(counts)

            # # 获取出现次数最多的值及其出现次数
            # most_common_value = unique_values[max_count_idx].item()
            # most_common_count = counts[max_count_idx].item()
            # gt_ins[b, i] = most_common_value


    # 维度处理
    if not use_batch_dim:
        gt_ins = gt_ins.squeeze(0)
        valid_counts = valid_counts[0] if len(valid_counts) > 0 else 0

    # 转换为numpy
    if return_numpy:
        gt_ins = gt_ins.numpy()
        valid_counts = np.array(valid_counts)

    return gt_ins, valid_counts

def convert_class_to_instance(gt_classes, instance_maps, max_instances, use_batch_dim=True, ignore_id=0, non_empty_maps=None):
    """类别转换（复用状态转换逻辑）"""
    return convert_state_to_instance(gt_classes, instance_maps, max_instances, use_batch_dim, ignore_id, non_empty_maps)

def convert_valid_map_to_instance(disp_valid_maps, instance_maps, max_instance, use_batch_dim=True, ignore_id=0):
    """
    returns:
        disp_valid_ins: numpy数组或torch张量 [bs, max_instances, seq, 1]
        valid_counts: 有效实例数数组
    """
    disp_valid_maps = _process_inputs(disp_valid_maps, use_batch_dim, 'valid_map')
    instance_maps = _process_inputs(instance_maps, use_batch_dim, 'instance_maps')
    return_numpy = isinstance(disp_valid_maps, np.ndarray)
    bs, seq_len, h, w, _ = disp_valid_maps.shape
    device = disp_valid_maps.device if hasattr(disp_valid_maps, 'device') else 'cpu'
     # 收集有效实例
    valid_counts = []
    all_masks = []
    instance_maps = instance_maps + 1
    for b in range(bs):
        instance_ids = torch.unique(instance_maps[b])
        valid_ids = instance_ids[instance_ids != ignore_id]
        valid_counts.append(len(valid_ids))

        batch_masks = []
        for ins_id in valid_ids:
            mask = (instance_maps[b] == ins_id) * non_empty_maps[b]
            batch_masks.append(mask)
        all_masks.append(batch_masks)

    # 初始化输出张量
    disp_valid_ins = torch.zeros((bs, max_instances, seq_len, 1), device=device)

    # 填充数据 # TODO: add valid pixel map
    for b in range(bs):
        for i, mask in enumerate(all_masks[b][:max_instances]):
            mask_4d = mask.unsqueeze(0).unsqueeze(-1).expand(seq_len, h, w, 1).float()
            masked_disp = disp_valid_ins[b] * mask_4d
            sum_disp = torch.sum(masked_disp, dim=(1,2))
            pixel_count = mask.sum()
            disp_valid_ins[b, i] = sum_disp / pixel_count if pixel_count > 0 else 0


    # 维度处理
    if not use_batch_dim:
        disp_valid_ins = disp_valid_ins.squeeze(0)
        valid_counts = valid_counts[0] if len(valid_counts) > 0 else 0

    # 转换为numpy
    if return_numpy:
        disp_valid_ins = disp_valid_ins.numpy()
        valid_counts = np.array(valid_counts)

    return disp_valid_ins, valid_counts


def convert_displacement_to_instance(gt_displacements, instance_maps, max_instances, use_batch_dim=True, ignore_id=0, \
                                     non_empty_maps=None, all_disp_valid_maps=None):
    """
    改进后的位移转换函数
    返回：
        gt_ins_disp: numpy数组或torch张量 [bs, max_instances, seq, 2]
        non_empty_maps: 非空maps[bs, H, W]
        valid_counts: 有效实例数数组
    """
    # 输入处理
    gt_displacements = _process_inputs(gt_displacements, use_batch_dim, 'displacements')
    instance_maps = _process_inputs(instance_maps, use_batch_dim, 'instance_maps')
    all_disp_valid_maps = _process_inputs(all_disp_valid_maps, use_batch_dim, 'valid_map')
    non_empty_maps = _process_inputs(non_empty_maps, use_batch_dim, 'non_empty_maps')
    return_numpy = isinstance(gt_displacements, np.ndarray)

    bs, seq_len, h, w, _ = gt_displacements.shape
    device = gt_displacements.device if hasattr(gt_displacements, 'device') else 'cpu'

    # 收集有效实例
    valid_counts = []
    all_masks = []
    instance_maps = instance_maps + 1
    for b in range(bs):
        instance_ids = torch.unique(instance_maps[b])
        valid_ids = instance_ids[instance_ids != ignore_id]
        valid_counts.append(len(valid_ids))

        batch_masks = []
        for ins_id in valid_ids:
            mask = (instance_maps[b] == ins_id) * non_empty_maps[b]
            batch_masks.append(mask)
        all_masks.append(batch_masks)

    # 初始化输出张量
    gt_ins = torch.zeros((bs, max_instances, seq_len, 2), device=device)

    # 填充数据 # TODO: add valid pixel map
    for b in range(bs):
        for i, mask in enumerate(all_masks[b][:max_instances]):
            mask_4d = mask.unsqueeze(0).unsqueeze(-1).expand(seq_len, h, w, 2).float()
            mask_non = non_empty_maps[:, None, ..., None]
            valid_mask = all_disp_valid_maps[b].unsqueeze(-1)
            # breakpoint()
            masked_disp = gt_displacements[b] * mask_4d * valid_mask * mask_non[b]
            # breakpoint()
            sum_disp = torch.sum(masked_disp, dim=(1,2))
            # TODO: the pixel counts
            pixel_count = (mask * valid_mask * mask_non[b]).sum()
            gt_ins[b, i] = sum_disp / pixel_count if pixel_count > 0 else 0

    # 维度处理
    if not use_batch_dim:
        gt_ins = gt_ins.squeeze(0)
        valid_counts = valid_counts[0] if len(valid_counts) > 0 else 0

    # 转换为numpy
    if return_numpy:
        gt_ins = gt_ins.numpy()
        valid_counts = np.array(valid_counts)

    return gt_ins, valid_counts
def create_one_hot(class_indices, num_classes):
    """
    将类别索引转换为one-hot编码。

    参数:
        class_indices: [height, width] 或 [bs, height, width] 形状的张量，
                       包含每个位置的类别索引。
        num_classes: 类别总数。

    返回:
        one_hot: [num_classes, height, width] 或 [bs, num_classes, height, width]
                 形状的one-hot编码张量。
    """
    # 获取输入张量的维度数
    dims = len(class_indices.shape)

    if dims == 2:
        # 没有batch维度的情况
        height, width = class_indices.shape
        # 初始化一个形状为[num_classes, height, width]的零张量
        one_hot = torch.zeros(num_classes, height, width, device=class_indices.device)
        # 使用scatter_函数进行one-hot编码
        one_hot.scatter_(0, class_indices.unsqueeze(0), 1)
    elif dims == 3:
        # 有batch维度的情况
        bs, height, width = class_indices.shape
        # 初始化一个形状为[bs, num_classes, height, width]的零张量
        one_hot = torch.zeros(bs, num_classes, height, width, device=class_indices.device)
        # 使用scatter_函数进行one-hot编码
        one_hot.scatter_(1, class_indices.unsqueeze(1), 1)
    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions (without or with batch dimension).")

    return one_hot

def generate_simulated_data(batch_size, height, width, num_state_cls, num_cls, seq_len, max_instances):
    # 初始化
    non_empty_maps = (torch.rand(batch_size, height, width) > 0.5).float()  # 非空掩码
    instance_maps = torch.zeros(batch_size, height, width).long()
    class_maps = torch.zeros(batch_size, height, width).long()
    state_maps = torch.zeros(batch_size, num_state_cls, height, width)
    disp_maps = torch.randn(batch_size, seq_len, height, width, 2)
    all_disp_valid_maps = (torch.rand(batch_size, seq_len, height, width) > 0.5).float()  # 位移有效掩码

    for b in range(batch_size):
        # 每个样本独立生成
        current_instance_id = 1
        for h in range(height):
            for w in range(width):
                if non_empty_maps[b, h, w] == 1:  # 如果是非空区域
                    # 随机决定是否为新实例
                    if torch.rand(1) > 0.5 or current_instance_id == 1:
                        instance_maps[b, h, w] = current_instance_id
                        current_instance_id += 1

                        # 随机分配类别
                        class_maps[b, h, w] = torch.randint(0, num_cls, (1,))

                        # 随机分配状态
                        state = torch.zeros(num_state_cls)
                        state[torch.randint(0, num_state_cls, (1,))] = 1
                        state_maps[b, :, h, w] = state

                        # 确保至少有一个有效的位移
                        valid_steps = torch.nonzero(all_disp_valid_maps[b, :, h, w]).squeeze()
                        if valid_steps.nelement() != 0:
                            disp_maps[b, valid_steps[0], h, w, :] = torch.randn(2)

    # 转换类别为one-hot编码
    class_gt = create_one_hot(class_maps, num_cls)
    state_gt = state_maps  # 假设state_maps已经是one-hot编码

    return non_empty_maps, instance_maps, class_gt, state_gt, disp_maps, all_disp_valid_maps

if __name__ == "__main__":
    # 定义变量
    batch_size = 2  # 单样本测试时，batch size设置为1
    height, width = 100, 100
    num_state_cls = 2
    num_cls = 5
    seq_len = 20
    max_instances = 8  # 根据实际情况调整，这里假设最大的实例数为8

    # 测试有batch维度的输入
    print("Testing with batch dimension:")
    # non_empty_maps, instance_maps, class_gt, state_gt, disp_gt, all_disp_valid_maps = generate_simulated_data(
    # batch_size, height, width, num_state_cls, num_cls, seq_len, max_instances)

    # non_empty_map
    non_empty_maps = torch.rand(batch_size, height, width)
    # 状态转换示例
    state_gt = torch.rand(batch_size, height, width, num_state_cls)  # 模拟状态数据
    # # 生成随机类别索引 [height, width]，取值范围从0到num_cls-1
    # state_indices = torch.randint(0, 2, (batch_size, height, width))

    # # 转换为one-hot编码
    # state_gt = create_one_hot(state_indices, 2)
    instance_maps = torch.randint(0, max_instances, (batch_size, height, width))  # 生成实例ID映射
    ins_state, _ = convert_state_to_instance(state_gt, instance_maps, max_instances=max_instances, use_batch_dim=True, \
                                             non_empty_maps=non_empty_maps)
    print("ins_state shape: {}".format(ins_state.shape))

    # 类别转换示例（逻辑与状态转换相同）
    class_gt = torch.rand(batch_size, height, width, num_cls)  # 模拟类别数据
    ins_class, _ = convert_class_to_instance(class_gt, instance_maps, max_instances=max_instances, use_batch_dim=True, \
                                              non_empty_maps=non_empty_maps)
    print("ins_class shape: {}".format(ins_class.shape))

    # 位移转换示例
    disp_gt = torch.randn(batch_size, seq_len, height, width, 2)  # 模拟位移数据
    all_disp_valid_maps = torch.randn(batch_size, seq_len, height, width)
    ins_disp, _ = convert_displacement_to_instance(disp_gt, instance_maps, max_instances=max_instances, \
                                                   all_disp_valid_maps=all_disp_valid_maps, use_batch_dim=True, \
                                                     non_empty_maps=non_empty_maps)
    print("ins_disp shape: {}".format(ins_disp.shape))

    print('Batch test finished!\n')

    # 测试无batch维度的输入
    print("Testing without batch dimension:")
    batch_size = 2  # 因为单样本，所以batch size仍然是1，但实际使用时不考虑这个维度
    non_empty_maps = torch.rand(height, width)
    # 状态转换示例

    state_gt_single = torch.rand(height, width, num_state_cls)  # 模拟单个样本的状态数据

    # # 生成随机类别索引 [height, width]，取值范围从0到num_cls-1
    # state_indices = torch.randint(0, num_cls, (height, width))

    # # 转换为one-hot编码
    # state_gt_single = create_one_hot(state_indices, num_cls)
    instance_map_single = torch.randint(0, max_instances, (height, width))  # 单个样本的实例ID映射
    ins_state_single, _ = convert_state_to_instance(state_gt_single, instance_map_single, max_instances=max_instances, \
                                                    use_batch_dim=False, non_empty_maps=non_empty_maps)
    print("ins_state_single shape: {}".format(ins_state_single.shape))

    # 类别转换示例（逻辑与状态转换相同）
    class_gt_single = torch.rand(height, width, num_cls)  # 模拟单个样本的类别数据

    # # 生成随机类别索引 [height, width]，取值范围从0到num_cls-1
    # class_indices = torch.randint(0, (height, width), num_cls)

    # # 转换为one-hot编码
    # class_gt_single = create_one_hot(class_indices, num_cls)

    ins_class_single, _ = convert_class_to_instance(class_gt_single, instance_map_single, max_instances=max_instances, \
                                                    use_batch_dim=False, non_empty_maps=non_empty_maps)
    print("ins_class_single shape: {}".format(ins_class_single.shape))

    # 位移转换示例
    disp_gt_single = torch.randn(seq_len, height, width, 2)  # 模拟单个样本的位移数据
    all_disp_valid_maps = torch.randn(seq_len, height, width)
    ins_disp_single, valid_counts = convert_displacement_to_instance(disp_gt_single, instance_map_single, max_instances=max_instances, \
                                                          all_disp_valid_maps=all_disp_valid_maps, use_batch_dim=False, \
                                                            non_empty_maps=non_empty_maps)
    print("ins_disp_single shape: {}".format(ins_disp_single.shape))
    print("valid counts shape:{}".format(valid_counts))
    print('Single sample test finished!')

# if __name__ == "__main__":
#     # 定义变量
#     batch_size = 2
#     height, width = 100, 100
#     num_state_cls = 2
#     num_cls = 5
#     seq_len = 20
#     max_instances = 8  # 根据实际情况调整，这里假设最大的实例数为8

#     # 状态转换示例
#     state_gt = torch.rand(batch_size, num_state_cls, height, width)  # 模拟状态数据
#     instance_map = torch.cat([
#         torch.randint(0, max_instances, (1, height, width)),
#         torch.randint(0, max_instances, (1, height, width))
#     ])  # 生成实例ID映射
#     ins_state, _ = convert_state_to_instance(state_gt, instance_map, max_instances=95)

#     print("ins_state shape: {}".format(ins_state.shape))

#     # 类别转换示例（逻辑与状态转换相同）
#     class_gt = torch.rand(batch_size, num_cls, height, width)  # 模拟类别数据
#     ins_class, _ = convert_class_to_instance(class_gt, instance_map, max_instances=95)
#     print("ins_class shape: {}".format(ins_class.shape))

#     # 位移转换示例
#     disp_gt = torch.randn(batch_size, seq_len, height, width, 2)  # 模拟位移数据
#     all_disp_valid_maps = torch.randn(batch_size, seq_len, height, width, 1)
#     ins_disp, _ = convert_displacement_to_instance(disp_gt, instance_map, max_instances=95, all_disp_valid_maps=all_disp_valid_maps)
#     print("ins_disp shape: {}".format(ins_disp.shape))

#     print('test finished!')
