# 出处与许可审计（P7 草案）

事实整理，不下法律结论；许可方案由用户决定。依据：`bevflow-library` skill 第四节。

## 1. 代码来源逐目录

| 路径 | 来源 | 许可/约束 |
|---|---|---|
| `bevflow/models/motionnet.py`、`datasets/motionnet.py`、`preprocess/{grid,convert,generate}.py`、`evaluators/motion_eval.py` | **MotionNet 派生**（MERL，逐行搬运+结构重构） | 上游 AGPL-3.0-or-later。衍生代码须与 AGPL 兼容 |
| `third_party/MotionNet/nuscenes-devkit/` | 上游自带 vendored nuScenes-devkit 子集 | 上游标注 CC-BY-NC-SA-4.0（非商业） |
| `bevflow/models/priormotion.py`、`vae.py`（内联）、`prior_extractor.py` | **用户自有代码派生**（旧 powerfulmotion，用户本人作品）；`prior_extractor.py` 五个类的源头亦为用户自写（文件头继承 MERL 版权声明——需用户确认其法律性质） | 用户决定；注意文件头沿用了 MERL 版权行，若类为用户原创需澄清 |
| `bevflow/models/priormotion.py` 的 `VpE`（论文重写版）、`RpE`（论文实现） | **按论文重新实现**（本次会话） | 论文代码未查阅；实现来自公式，原创性较强（仍受论文方法本身约束，非代码许可问题） |
| `bevflow/models/legomotion/` | **按论文从零实现**（代码已失，无旧代码） | 完全原创实现 |
| `bevflow/trainers/`、`scripts/`、`evaluators/viz.py`、`utils/` | 本次会话原创（部分逻辑参考上游流程） | 原创 |
| `min_norm_solvers.py` 等 MGDA、`third_party/` 全部 | 上游，**不随库发布**（gitignore） | AGPL（若发布则须兼容） |
| 第三方依赖：torch、numpy、scipy、sklearn、matplotlib、pyyaml、tensorboard、numba、pyquaternion、Pillow、opencv | PyPI 标准包 | 各自许可（BSD/MIT/Apache/LGPL 混合，均兼容 AGPL 之外的分发） |

## 2. 已知约束（来自项目笔记，待核对）

1. MERL 版 MotionNet：AGPL-3.0-or-later；其自带 nuScenes-devkit 文件标 CC-BY-NC-SA-4.0。
2. 凡是从 MotionNet 复制或改写的代码（上一表第 1 行），库的许可须与之兼容。
3. nuScenes 数据：项目笔记记"未核实"，社区通行 CC BY-NC-SA 4.0（非商业）。以此数据训练的权重，发布前须核对条款。
4. 论文 PDF 与旧代码属 `ref/`，gitignore，不发布。

## 3. 可选方案（供用户决定，agent 不代选）

| 方案 | 内容 | 影响 |
|---|---|---|
| A 全库 AGPL-3.0 | 整个 bevflow 采用 AGPL-3.0-or-later | 最简、合规；但下游商用受限（学术场景通常可接受） |
| B 派生部分隔离 | 核心库用宽松许可（如 Apache-2.0），MotionNet 派生模块单独成可选子包并标 AGPL | 增加包结构复杂度；用户主方法（PriorMotion/LEGO 部分）可用宽松许可 |
| C 共享部分按论文重写 | 把 MotionNet 派生的预处理器/加载器/评测器按论文与公开协议重写 | 工作量 1-2 周；换来许可自由度 |

## 4. 发布前动作（用户执行，agent 准备）

1. 用户选定许可方案后：调整 LICENSE、各文件头。
2. `scripts/audit_release.sh` 私有信息扫描（绝对路径/用户名/主机名/token/大文件）——待写。
3. 不公开仓库、不上传权重，直到用户明确决定。
