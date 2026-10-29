# Changelog

## v0.1 (待发布)

首个版本：统一管线上的 MotionNet 基线复现 + PriorMotion 逆向复现（DSpG / RVpE）+ LEGO-Motion 按论文实现。

- 数据：nuScenes → MotionNet 协议预处理（确定性移植，17065/1719/4309 序列），数据卡与统计。
- 模型：MotionNet（官方权重可直接加载）；PriorMotion（DSpG 条件 VAE + SGRU；RVpE 按论文实现）；LEGO-Motion（IaIE + IeBE + 掩码头）。
- 评测：motionnet_v1 协议（三档速度、1.0s 单点口径）+ 扩展口径（空间分桶、径向/切向分解、实例稳定性）。
- 训练：统一 trainer（train.jsonl、run_manifest、resume auto、周期性 ckpt）。
- 工具：model zoo 生成、发布前私有信息扫描、自动恢复看守。
