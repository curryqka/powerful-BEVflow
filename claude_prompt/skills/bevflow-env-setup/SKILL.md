---
name: bevflow-env-setup
description: 在集群上勘察机器，并用 uv 按 MotionNet 上游仓库配置 Python 环境时使用。覆盖勘察，克隆并固定上游，uv 虚拟环境，补丁协议，验证与锁文件。
---

# 目标

得到一个能跑 MotionNet 的 uv 环境，别人能复现。上游代码保持原样，改动写成补丁。

# 一、勘察，只读

```bash
hostname; nvidia-smi; nvcc --version 2>/dev/null
uv --version; python3 --version
command -v sinfo squeue sbatch && sinfo -o "%P %a %l %D %G" | head -20
df -h "$HOME" "$NAS"
curl -sI --max-time 10 https://github.com | head -1
curl -sI --max-time 10 https://zenodo.org | head -1
```

输出整理进 `docs/setup/recon.md`。要回答的问题：

- 有调度器吗？分区，时限，GPU 类型。没有就记下哪些卡空闲。
- GPU 计算能力是多少？它决定 torch 与 CUDA 版本。
- 网络通哪些站点？不通，就列出要我上传的文件。
- uv 是否已装？没装就问我，不要自己装到别处。

# 二、克隆并固定上游

1. 克隆 `https://github.com/merlresearch/MotionNet` 到 `third_party/MotionNet`。
2. 项目笔记说作者页 `pxiangwu/MotionNet` 只有 README。克隆后两个仓库都看一眼，以含数据生成，训练，评测脚本的为准。
3. 记录 commit，license，README 的环境要求，以及这几件事对应的脚本：生成数据，训练 STPN 主模型，评测位移，评测分类。写进 `docs/setup/upstream.md`。

克隆不通就问我，不要换镜像域名。

# 三、用 uv 建环境

环境前缀放 `$EXP/_env/bevflow`。uv 缓存放 `$EXP/_cache/uv`，避免占满 `$HOME`。

```bash
export UV_CACHE_DIR=$EXP/_cache/uv
uv venv $EXP/_env/bevflow --python 3.10
source $EXP/_env/bevflow/bin/activate
```

本机全局 `UV_CACHE_DIR` 指向无权限的 `/media/nas_users/uv`。每个新 shell 都要先执行上面的 export，再跑 uv，否则 uv 直接报权限错误。

安装用 `uv pip install`：

1. 先装 torch。版本由 GPU 计算能力和驱动决定，不照搬上游的老版本。较新的 uv 支持 `--torch-backend=auto`，先用 `uv pip install --help` 确认。不支持就指定 `--index-url https://download.pytorch.org/whl/cuXXX`。
2. 再装 numpy 等科学计算库，以及上游 requirements 里 nuscenes-devkit 之外的包。nuscenes-devkit 用上游 vendored 的精简版（`third_party/MotionNet/nuscenes-devkit`，PYTHONPATH 引用），不装 pip 版。
3. MGDA 相关文件（`min_norm_solvers.py`，`train_*_MGDA.py`）只在 PyTorch 1.1 能跑，本项目不用。相关的兼容报错不修，记一行跳过。
4. 每装一步就验证，不要最后一起验证。

# 四、补丁协议

- 不直接改上游目录。需要改时，用 `diff -u` 生成 `patches/NNNN-<name>.patch`，在 `docs/setup/upstream_patches.md` 记一行：改了什么，为什么。
- 常见需要补丁的地方：`np.int` 与 `np.float` 已移除，`torch.load` 默认参数变化，过时的 DataLoader 参数，路径硬编码。
- 补丁只修兼容，不改算法。拿不准算不算算法改动，问我。

# 五、验证

```bash
python - <<'EOF'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda"); print((x @ x).sum().item())
import nuscenes; print("nuscenes ok")
EOF
```

再 import 上游的模型与数据模块，并用 mini 元数据实例化一次 `NuScenes`。

# 六、锁文件

提交 `requirements.lock.txt`，由 `uv pip freeze` 生成，并在 `docs/setup/env.md` 写下重建环境的命令。

遇到装不上的包，先看报错，再决定换版本还是写补丁。判断不了就问我。

# 七、官方权重

下载 MotionNet 官方权重，Zenodo 记录 8215963。先打开页面确认文件名与大小，下载到 `$EXP/_models/motionnet_official/`，记录 sha256 进 `docs/setup/upstream.md`。网络不通就问我，让我上传。
