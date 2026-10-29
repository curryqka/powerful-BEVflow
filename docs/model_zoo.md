# Model Zoo（自动生成，不手改）

由 scripts/make_model_zoo.py 生成，git 9f72e8b4fc0b43846c6950f6da3b6d9512a6e4e9

| run_id | 模型 | epe static | slow@1.0s | fast@1.0s | OA | MCA | ckpt | 说明 |
|---|---|---|---|---|---|---|---|---|
| official_test_bs1 | MotionNet（官方权重） | 0.0145 | 0.2467 | 1.0110 | 0.9632 | 0.7127 | model.pth | Zenodo 8215963，test |
| PM_DSpG_s0_ep45_test | PriorMotion DSpG s0 | 0.0311 | 0.4811 | 0.8605 | 0.9402 | 0.7674 | epoch_45.pth | Table 7(e) 对照臂 |
| PM_DSpG_s1_ep45_test | PriorMotion DSpG s1 | 0.0168 | 0.2351 | 0.9150 | 0.9568 | 0.7349 | epoch_45.pth | Table 7(e) 对照臂 |
| PM_rvpe_s0_test | PriorMotion RVpE s0 | 0.0157 | 0.2368 | 0.9149 | 0.9606 | 0.7226 | epoch_45.pth | 全配置，Table 2 对照 |
| PM_rvpe_s1_test | PriorMotion RVpE s1 | 0.0164 | 0.2324 | 0.8982 | 0.9595 | 0.7260 | epoch_45.pth | 全配置，Table 2 对照 |
