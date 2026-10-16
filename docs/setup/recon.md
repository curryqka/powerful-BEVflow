# 勘察记录

日期：2026-10-16。主机：`dsw-1118939-54b9857b67-fns5l`（阿里云 DSW 容器）。

## 调度器

无。`sinfo`，`squeue`，`sbatch` 都不存在。长任务用 nohup + tmux，模板见 `scripts/slurm/`（保留空模板备用）。

## GPU

2×RTX 4090，各 46068 MiB，勘察时全部空闲（0%，1 MiB）。驱动 580.95.05，支持到 CUDA 13.0。计算能力 sm_89，需要 cu118 及以上的 torch 轮子。容器内无 nvcc，torch 用自带 CUDA 运行时的轮子即可。

## 磁盘

| 挂载点 | 容量 | 可用 |
|---|---|---|
| `/`（含 `$HOME`） | 98G | 91G |
| `/media/nas_users`（NAS） | 1.0P | 906T |

## 网络

github.com 与 zenodo.org 均可达（HTTP 200）。pypi 走 uv 默认源。

## uv 与 Python

uv 0.13.1。系统 python 3.12.3。项目环境用 uv 托管的 python 3.10，前缀 `$EXP/_env/bevflow`。

注意：全局环境变量 `UV_CACHE_DIR=/media/nas_users/uv` 指向 root 所有的目录，普通用户不可写。每个 shell 必须先 `export UV_CACHE_DIR=$EXP/_cache/uv`，否则 uv 直接报权限错误。

## 原始 nuScenes

`/media/nas_users/kaneqian/data/Nuscences`（目录名拼写如此）。含 `v1.0-trainval`，`v1.0-test` 元数据，`samples`，`sweeps`，`maps`，can_bus。无 v1.0-mini 原始数据。

## 决定（常规工程决定，记录备查）

- 冒烟的"mini 划分"不用 nuScenes mini 原始数据（本机没有），从上游 train split 抽 10 个场景（6/2/2）。`v1.0-mini` 元数据加载验证用 `v1.0-trainval` 元数据代替。
- NAS 各根下无同名 `bevflow` 目录冲突，已新建目录树。
- git 身份在本仓库内设为 `kaneqian <kaneqian@dsw.local>`，如需改全局身份告诉我。
