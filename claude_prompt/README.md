# BEVflow 集群部署包：使用说明

## 内容

```
prompts/00_master_prompt.md           总控提示词，贴给 agent 当第一条消息
skills/bevflow-workspace-layout/      目录与路径，两个存储区，隔离
skills/bevflow-env-setup/             勘察，上游仓库，uv 环境，补丁协议
skills/bevflow-data-pipeline/         nuScenes 链接，划分，数据生产，数据卡
skills/bevflow-smoke-test/            五步冒烟，可视化数值检查
skills/bevflow-refactor/              五模块重构，行为不变的测试
skills/bevflow-experiment-run/        参数卡，两套配置，官方权重，第一轮
skills/bevflow-reverse-reproduce/     第六阶段，你触发后才用
skills/bevflow-docs-report/           文档分类，报告模板
```

## 部署

1. 把 `skills/` 下的八个目录复制到集群的 `~/.claude/skills/`。仓库建好后，也可以放进 `$CODE/.claude/skills/`。
2. 把 `prompts/00_master_prompt.md` 整体贴给 agent，作为第一条消息。
3. 让 agent 建好仓库后，把总控提示词的第 0 节与第 3 节写进仓库根的 `CLAUDE.md`。第 3 节含你的开发规范，每个会话都会重新读到。

## 这一版的取向

- 自动推进。检查点通过就进入下一阶段，不等确认。agent 默认一路做到 P5。
- 只在"会影响结果解释，或事后难以回退"的事情上提问，一次问一批。常规工程决定自己定。
- 参考值对照是对照，不是门。对不上就排查并写进报告，不卡住后面的工作。
- 环境用 uv：`uv venv` 加 `uv pip install`，前缀与缓存都在 `$EXP` 下。
- 基线现状按实际写：只有 MotionNet 官方权重，需要下载。PriorMotion 与 LEGO-Motion 在 P6 复现后训练。BE-STI 只引论文数字。

## 开发规范在哪里

全部在总控提示词第 3 节：

- 注释，docstring，异常（assert 与不 catch 致命错误）。
- 删除文件需确认，脚本里的删除默认只列出，必须显式 flag。
- 长输出先写入 `$CACHE/cmdlogs/` 再 head 或 tail。
- `mv`，`cp`，`rm` 的批量操作，先打印命令再执行。

"使用 Claude 的要求"那条，不要直接 accept edits 去做复杂文件操作，是给你自己的操作习惯，agent 不需要执行。agent 侧对应的是上面最后一条。

## 默认值，可改

| 项目 | 默认 | 改哪里 |
|---|---|---|
| 项目二级目录名 | `bevflow` | 总控提示词第 0 节 |
| 代码区 | `/home/kaneqian/BEVflow` | 同上 |
| NAS 根 | `/media/nas_users/kaneqian` | 同上 |
| 实验区与结果区的分工 | 实验区放训练 run，结果区放评测与可视化 | skill `bevflow-workspace-layout` |
| uv 环境位置 | `$EXP/_env/bevflow` | skill `bevflow-env-setup` |
| 上游仓库 | MERL 版 MotionNet，克隆后与作者页对照 | skill `bevflow-env-setup` |

## 你可能要提供的输入

| 输入 | 什么时候要 |
|---|---|
| 调度器分区与账号 | P0，需要时 agent 会问 |
| 旧项目代码路径，以及可用的日志与 ckpt | P6 |

## 2026-10-16 修订

- `$CODE` 原有的上游克隆已清空（保留本目录），P1 由 agent 按 skill 重新克隆。
- 原始 nuScenes 路径写进总控提示词第 0 节：`/media/nas_users/kaneqian/data/Nuscences`。split 直接用上游 `data/split.npy`（已核实正好 500/100/250）。
- 官方权重下载从 P4 提前到 P1，P3 重构的行为不变测试要用它。
- 本机全局 `UV_CACHE_DIR` 指向无权限路径，env-setup skill 已加提示：每个 shell 都要先 export。
- MGDA 相关文件明确不迁移不打补丁；nuscenes-devkit 以上游 vendored 版为准，不装 pip 版。
