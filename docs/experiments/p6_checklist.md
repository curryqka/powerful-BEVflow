# P6 准备清单（逆向复现 PriorMotion / LEGO-Motion）

触发条件：用户说"开始第六阶段"并给出旧项目代码路径。此前本清单只盘点不动手。

## 需要用户提供

1. 旧项目代码路径（本地或 github.com/curryqka/powerful-BEVflow 的 PriorMotion/MotionNet 分支，需确认以哪个为准）。
2. 旧项目留下的训练日志、中间 ckpt、论文报告数字——旧 ckpt 加载到新模型做同输入对比是最强的等价验证。
3. PriorMotion / LEGO-Motion 各自的三档 EPE 报告值（作为复现对照）。

## 已就绪的基础

- 数据：motionnet_v1 已产出并断言（17065/1719/4309）；PriorMotion 若需额外预处理产物，生成到新版本目录，不改 motionnet_v1。
- 模块接口：`docs/design/interfaces.md`——models/losses 的 registry、evaluator 的协议版本位、trainers 的 resume 机制。
- 评测：motion_eval 的指标注册方式、metrics.json 的 meta.eval_protocol 字段。
- 塌缩教训：含一致性/先验/ KL 类辅助损失的配方，训练首 epoch 必须监控辅助损失与 disp 分项（判据见 docs/reports/R1.md 第 5 节）；PriorMotion 的 KL 先验损失属同类风险。

## 需要在 P6 内决定（到时问用户）

1. 统一口径：旧管线与 motionnet_v1 的数据/评测差异取哪套（reverse-reproduce skill 第一节）。
2. 复现顺序：PriorMotion 先（方案里第一个实验依赖它的接口）还是 LEGO-Motion 先（与 MotionNet 同主干，搬运成本低）。
3. 训练预算沿用 R1B（bs4，45 epoch）还是按旧项目原设置。

## 风险登记

- 旧代码若也是 torch 旧版写成，numba/np 别名/grid_sample 语义三类坑会重现，patches 协议已验证可覆盖前两类。
- SGRU 自回归解码与实例 query 在线提取的计算图较复杂，逐模块移植时保持"前向形状与数值合理性检查"节奏。
