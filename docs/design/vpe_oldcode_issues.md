# VpE 旧代码问题清单（具体到行）

对照基准：PriorMotion 论文 p.4 §3.2.2 与式(5)。旧代码 = `ref/priormotion/code/` 下
`data/instance_extractor.py`（get_instance_infoV2）与 `model/prior_extractor.py`（GridInsTransformer）。

## 论文规格（复核依据）

- 实例序列：**沿轨迹**采样 N 个网格点（"sample N grid points (xi, yi) **along its trajectory**, analogous to SD map polyline extraction"，式5 前文）。
- 每点三种特征：正弦位置编码、实例相关位移向量、one-hot 类别。
- 时序：**位移的时间进化 M_I ∈ R^(T×2) 由 LSTM 编码**。
- 维度核算：D_I = N·d_pos + N_C + **T·d_M**。

## 重大问题（方法论层，三处）

### 1. 没有时序建模，LSTM 完全缺失

- 位置：`instance_extractor.py:279` `last_disp_field_gt = all_disp_field_gt[:, -1, ...]` —— 整个函数只取**末帧**位移场，之后所有位移插值（:333-345）只用末帧。
- 位置：`prior_extractor.py:99-182` GridInsTransformer —— **没有 LSTM**。
- 后果：式(5) 的 T·d_M 项不存在；"位移的时间进化"被压缩成一张末帧静态图。所谓 vector prior 实际是"形状 + 末帧运动"，不是"轨迹先验"。

### 2. 采样的是形状 footprint，不是轨迹

- 位置：`instance_extractor.py:313-322` —— 取实例**当帧占据格子** → ConvexHull → 在 hull 的包围盒上 `np.linspace` 建规则网格（:321-322）。采样点 = 包围盒内的均匀网格（对形状重采样），与运动轨迹无关；T 帧的运动只通过末帧位移数值进入。
- 后果：论文的"SD 折线式轨迹采样"被替换成"footprint 网格采样"。VpE 无法表达运动方向的时间结构（加速/减速、转弯弧线），而这正是矢量先验相对栅格先验的立身之本。

### 3. 维度核算错误：时间维被当成点维

- 位置：`en_model.py:739`（`num_grid` 默认 100 见 :719，`train_VAE.py:646` 实传 100）：`input_dim = 32*num_grid + 32*num_grid + 5`。`prior_extractor.py:99-182` 是消费方，不含此维度计算。
- 解读：第一个 32×N = 位置编码（每点 2→32）；第二个 32×N = **每点末帧位移**（2→32 线性）；+5 类别。
- 论文：第二项应为 T·d_M（时间步 × LSTM 输出），不是 N×32（点 × 位移嵌入）。原型把"每个空间采样点"当成了"每个时间步"。

## 次级问题

### 4. griddata 的用法浪费且丢信息
- 位置：`instance_extractor.py:326-331` `griddata(points=hull_points, values=hull_points, ...)` —— 把 hull 顶点坐标同时当插值点和值，实质只是"在 hull 顶点间线性插值坐标"（生成规则网格）。
- 位置：`:333-345` 位移插值只用 **hull 顶点**的末帧位移值，实例内部格子的位移被丢弃（内部点运动信息丢失）。

### 5. 实例特征对点顺序敏感
- 位置：`prior_extractor.py:163` `graph_polylines.view(npolylines, npoints * pdim)` —— N 个点展平成 6405 维（num_grid=100，`train_VAE.py:646`）喂线性层，点序变化即特征变化；无置换不变性设计。

### 6. 填充点是伪数据
- 位置：`instance_extractor.py:357-370`（`np.pad(..., mode='constant')` 坐标补零到 (0,0)）与 `:331/:338/:345`（fill_value=0）、`:354-355`（nan_to_num）——补出来的点在位置编码里变成原点的真实嵌入，给每个不足 N 点的实例注入"原点伪点"。

### 7. 类别取单格
- 位置：`instance_extractor.py:376` `category = pixel_cat_maps_gt[b, :, positions[0][0], positions[0][1]]` —— 用第一个格子的类别代表整个实例（实例内部混类时不可靠）。

### 8. 退化情形
- 位置：`:308-309、:350-351` 单点/退化 hull → `np.tile(positions[0], (num_grid,1))` —— N 个完全相同的点，位置编码退化为常数，但仍进入 transformer。

## 对旧结果的含义

- 旧实验里的 "VpE" 实际是"footprint + 末帧位移"先验。它仍能带来小幅提升（旧论文 Table 7(a)：fast 0.9733→0.9580），因为末帧位移本身有益；但论文宣称的时序运动先验从未被实现。这解释了为什么 VpE 单独作用的消融增益（1.57%）明显弱于 RpE（3.92%）——弱的不是"矢量先验"这个想法，而是实现。
- 本复现（D9，用户裁决）：VpE 按论文重写 —— 轨迹等弧长采样 N 点 + 逐帧位移序列经 LSTM + D_I = N·d_pos + N_C + T·d_M；PCA 与融合点沿用原型（判定合理）。

> 2026-10-23 复核：条目结论全部成立（含论文原文逐句核对：p.4 "sampled along its trajectory"；"M_I ∈ R^(T×2) ... captured through an LSTM layer"；Table 7(a) 0.9733→0.9580=1.57%，RpE 0.9733→0.9351=3.92%）。行号已按 `ref/priormotion/code` 副本实测修正；训练路径确认走 `get_instance_infoV2`（train_VAE.py:642）且 num_grid=100（:646）。`prior_extractor.py` 全文件无 `nn.LSTM/nn.GRU`（全仓 ref 代码 grep 零命中）。
