# 环境重建

uv 0.13.1，python 3.10（uv 托管）。环境前缀 `$EXP/_env/bevflow`，uv 缓存 `$EXP/_cache/uv`。

```bash
export EXP=/media/nas_users/kaneqian/Experiment/bevflow
export UV_CACHE_DIR=$EXP/_cache/uv   # 全局 UV_CACHE_DIR 指向无权限路径，必须先覆盖

uv venv $EXP/_env/bevflow --python 3.10
uv pip install --python $EXP/_env/bevflow/bin/python --torch-backend=auto torch
uv pip install --python $EXP/_env/bevflow/bin/python \
    pyquaternion matplotlib numpy opencv-python-headless tqdm Pillow \
    numba scikit-learn cachetools
uv pip freeze --python $EXP/_env/bevflow/bin/python > requirements.lock.txt
```

装好的版本（2026-10-16）：torch 2.14.1+cu132，numpy 2.2.6，numba 0.68.0，scikit-learn 1.7.2，opencv-python-headless 5.0.0.93。完整清单见仓库根 `requirements.lock.txt`。

## 与上游 requirements.txt 的差异

上游清单漏了三样，实际 import 时补上：

- `numba`：`data/data_utils.py` 用 `@njit`。
- `scikit-learn`，`cachetools`：vendored nuscenes-devkit 需要。

`nuscenes-devkit` 不装 pip 版，用上游 vendored 精简版，运行时加：

```bash
export PYTHONPATH=$CODE/third_party/MotionNet:$CODE/third_party/MotionNet/nuscenes-devkit/python-sdk
```

cv2 装 `opencv-python-headless`（服务器无 GUI，import 兼容）。

## 验证结果

- torch CUDA：`2.14.1+cu132 True RTX 4090`，1024² 矩阵乘正常。
- 上游模块 import：`model`，`data.nuscenes_dataloader`（`TrainDatasetMultiSeq`，`DatasetSingleSeq`），vendored `nuscenes` 全部通过。
- NuScenes trainval 元数据加载：850 scenes，34149 samples。
