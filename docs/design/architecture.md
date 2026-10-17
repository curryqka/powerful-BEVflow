# 架构

## 模块与依赖方向

```
scripts/（薄 CLI：train.py eval.py gen_data.py viz.py）
   ↓
trainers ──→ evaluators ──→ datasets
   ↓             ↓              ↓
models（网络，损失）  preprocess（grid/convert/generate）
   └──────────── utils（paths, seed, config, registry）
```

依赖只向下，无环。新方法只需换 models 里的模块（registry 按名构建）。

## 数据流

```
原始 nuScenes（只读链接 $DATA/raw/nuscenes）
  → preprocess.generate（按 split 抽帧，每场景产出 <scene>_<seq>/<adj>.npy 稀疏 dict）
  → eval 根目录拍平（scripts/flatten_eval_data.py，DatasetSingleSeq 只认扁平文件）
  → datasets（稀疏 dict 展开成稠密张量，含动静分类 GT）
  → models（STPN 主干 → 位移/类别/动静三头）
  → trainers（加权 SmoothL1 + 一致性损失，train.jsonl 记录）
  → evaluators（三档 EPE + OA/MCA，metrics.json，motionnet_v1 协议）
```

## 关键约定

- **路径唯一来源**：`bevflow/utils/paths.py`，环境变量 BEVFLOW_* 覆盖，缺省值内置。写文件入口走 `writable_path` 断言。
- **state_dict 键名**：模型属性名与上游一致（module.stpn.* 等），官方权重直接加载（golden 测试保证）。
- **张量布局**：位移预测内部为 (B, F, 2, h, w)；数据集侧 GT 为 (F, h, w, 2)。两者转换点已在代码注明。
- **协议常量**：网格 256×256×13，体素 0.25m，范围 ±32m，动静阈值 0.2 m/s，三档 0/5/20 m/s，ROI 裁边 8 像素。改动任何一项都意味着新协议版本。
- **配置**：YAML + `_base_` 继承 + `--set` 点号覆盖。解析后的完整配置存 run 目录。

## 扩展点（最小集）

- `bevflow/models` registry：按名字构建 backbone/head（换解码头只改配置）。
- losses：LossConfig 加字段或新函数注册。
- metrics：motion_eval 是 motionnet_v1 协议；新指标在 evaluators 里加函数，不动旧协议。
