# 中间文件台账

用途：你后面清理时的清单。新增产物随建随记。大小为 2026-10-18 快照。

## 核心资产（保留）

| 路径 | 大小 | 内容 |
|---|---|---|
| `$DATA/preprocessed/motionnet_v1/` | 54G | 全量预处理数据（17065/1719/4309）+ data_card + data_stats。一切训练的输入 |
| `$DATA/splits/` | <1M | split.npy（500/100/250）+ mini_split.npy |
| `$DATA/golden/` | ~140M | golden 基线（forward.npz/loss_inputs.npz），移植对拍用 |
| `$DATA/raw/nuscenes` | 链接 | 原始数据只读链接 |
| `$EXP/_env/bevflow/` | ~6G | uv 环境 |
| `$EXP/_models/motionnet_official/` | 310M | 官方权重（zip+解压） |
| `$EXP/R1B/<seed>/ckpt/` | ~2.9G/run | R1B 三种子 ckpt（epoch 1,5,10-45 + last.pt）。主表资产 |
| `$EXP/R1A/20261017_R1A_motionnet_s1/ckpt/` | ~2.9G | R1A 官方结果（种子 1） |
| `$EXP/P6PM/` | （增长中） | PriorMotion 参考实现双种子 |
| `$RESULTS/`（registry/R1/eval/viz/_smoke） | ~2G | 全部评测产物与可视化 |
| `$CODE/ref/` | 15M | P6 材料（论文/旧代码链接/notes），gitignore |

## 过程产物（小，保留至对应阶段收尾）

| 路径 | 大小 | 内容 | 处置建议 |
|---|---|---|---|
| `$EXP/_cache/cmdlogs/` | ~10M | 全部长命令日志（审计线索） | 保留 |
| `$EXP/_cache/papers/` | ~1M | 两篇论文的文本抽取（P6 工作材料） | P7 开源审计后可删 |
| `$EXP/_data_logs/` | ~5M | 数据生产日志 | 保留 |
| `$EXP/_cache/smoke_cwd/`、`upstream_rerun_cwd/` | <1M | 冒烟/审计用 scratch（split npy 副本） | **现在可删** |
| `/tmp/upstream_bs8_1ep/` | ~1G | 上游训练器仲裁实验产物（已完成使命） | **现在可删** |
| `/tmp/eval_debug/`、`/tmp/eval_probe/`、`/tmp/upstream_rerun/` | ~1G | 塌缩调查/可视化调试临时物 | **现在可删** |
| `$EXP/R1A/20261017_R1A_motionnet_s0/` | ~2.9G | 塌缩的种子 0 run（塌缩调查实证材料） | R1 报告归档后可删 |
| `$EXP/R1A/20261018_R1A_official_s0/` | <1M | 已取消的重定义 run | 已终态，可删 |
| `$EXP/P6PM/20261018_pmcheck_s0/` | <1M | 接线验证跑（被 timeout 杀） | 已终态，可删 |
| `$EXP/R1B/*/logs/tb/` | 各~100M | tensorboard 事件文件 | 不需要看曲线时可删 |
| `$EXP/P6PM/2026101{8,20}_PM_*` | 各~2.9G（终态后） | PriorMotion 忠实双种子 + kld 探索双变体 | 报告归档后按 zoo 引用裁剪 |
| `$EXP/_cache/watchdog.log`、`watchdog_runs.txt`、`seq_listings/` | <1M | 看守日志/规格、目录列表缓存 | 保留（看守运行依赖） |
| `$EXP/_cache/cmdlogs/watchdog_*.log` | 各<1M | 看守重启的训练 stdout | 保留 |

## 卫生规则（即刻生效）

1. 新增的中间产物目录在创建时登记进本表。
2. `$EXP/_cache/` 与 `/tmp` 是"可牺牲区"：任何清理从这里开始，不影响可复现性（真正的资产在 $DATA/$RESULTS/各 run 目录/仓库）。
3. run 目录的清理原则：model zoo 定稿（P7）后，仅保留 zoo 引用的 ckpt 与 last.pt。

## 暂停状态（2026-10-25 起，rebuttal 占用双卡；**2026-10-26 07:24 已恢复**）

- 恢复结果：fuser（e20 断点→续跑）、rvpe_s0（e19→续跑）在 GPU0；rvpe_s1 直接上 GPU1 续跑。

- 训练已全部停止：fuser_s0（e17 断点）、rvpe_s0（e16 断点，主线），进度由 latest.pt 保全。
- 看守与排队器已停；原规格备份在 `$EXP/_cache/watchdog_runs.txt.suspended`。
- **恢复步骤（用户说"用完"后）**：
  1. `cp $EXP/_cache/watchdog_runs.txt.suspended $EXP/_cache/watchdog_runs.txt`
  2. `setsid nohup bash $CODE/scripts/watchdog.sh &`（看守会自动发现未完成 run 并 resume）
  3. `setsid nohup bash $CODE/scripts/queue_rvpe_s1.sh &`（fuser+rvpe_s0 结束后自动接 rvpe_s1 并注册看守）

## 暂停状态二（2026-10-26 18:xx，用户 rebuttal 第二轮，立即停机）

- fuser_s0（e25 中段）与 rvpe_s0（e24 中段）已立即停止；latest.pt 保全（最多损失 ~500 步 ≈ 4 分钟）。
- rvpe_s1（e14 断点）此前已停。看守/排队器/用户的收尾停止器均已停。
- 恢复命令（resume auto）：
  `CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/exp/priormotion_fuser.yaml --set run.run_id=20261023_PM_fuser_s0 train.seed=0 train.val_interval=2`
  `CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/exp/priormotion_rvpe.yaml --set run.run_id=20261023_PM_rvpe_s0 train.seed=0 train.val_interval=2`
  `CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/exp/priormotion_rvpe.yaml --set run.run_id=20261023_PM_rvpe_s1 train.seed=1 train.val_interval=2`
  （恢复优先级已修复：同 epoch 时用 epoch_N.pth 而非 mid-epoch latest.pt）
