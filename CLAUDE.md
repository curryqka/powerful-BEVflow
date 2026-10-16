# BEVflow 会话规则

本文件从总控提示词 `claude_prompt/prompts/00_master_prompt.md` 的第 0 节与第 3 节原样拷入。每个会话先读它。

## 0. 变量

要改路径，只改这一节。

```
PROJECT = bevflow
CODE    = /home/kaneqian/BEVflow
NAS     = /media/nas_users/kaneqian
EXP     = $NAS/Experiment/bevflow
RESULTS = $NAS/results/bevflow
DATA    = $NAS/data/bevflow
CACHE   = $EXP/_cache
RAW_NUSCENES = /media/nas_users/kaneqian/data/Nuscences
```

## 3. 规则

### 边界

1. 只写 `$CODE`，`$EXP`，`$RESULTS`，`$DATA`。
2. `/home` 不放大文件。代码区里的 `data`，`experiment`，`results` 是软链接，写进 `.gitignore`。
3. 删除文件要敬畏。除了删你自己生成的临时文件，其他删除都先问我。脚本里的删除，默认只列出，必须显式 flag 才真删。
4. `mv`，`cp`，`rm` 的批量或带逻辑的操作，先打印将要执行的命令，确认无误再执行。
5. `third_party/MotionNet` 是上游，只读。要改，写成 `patches/` 下的补丁，并注明原因。
6. 每个数字来自日志或 `metrics.json`，不凭记忆写，不编造。
7. 不用 sudo。共享集群上，登录节点不跑重活，占卡前先看 `nvidia-smi`。

### 不确定就问我

不确定的地方多问，一次问一批。判断标准：这个选择会影响结果的解释，或者事后难以回退，就算不确定。例如 split 的来源，统一用哪套数据管线，上游代码里语义不清的参数。常规的工程决定自己定，在文档里记一行。

问的同时，继续做不依赖答案的部分。

### 代码规范

1. 注释：一眼能懂的代码不写注释。只有跨越多行才能看懂的逻辑才写。
2. docstring：不过度写。多数函数不写，更不要写会抛什么异常。
3. 异常：输入有预期，且上游逻辑已保证的，用 `assert`。严重且无法恢复的错误，不能 catch，要让它中断程序。禁止 `try: ... except Exception as e: print(...)` 这类写法。
4. 删除：见上面的边界第 3 条。

### 命令输出

输出预计很长的命令，先重定向到 `$CACHE/cmdlogs/<name>.log`，再用 `head` 或 `tail` 查看。不要为了看输出而重复运行命令。

## 环境备忘

- 每个 shell 跑 uv 前：`export UV_CACHE_DIR=$EXP/_cache/uv`（全局变量指向无权限路径）。
- uv 环境前缀：`$EXP/_env/bevflow`，python 3.10。
- nuscenes-devkit 用上游 vendored 版：`PYTHONPATH` 加 `third_party/MotionNet/nuscenes-devkit/python-sdk`。不装 pip 版。
- 本机无调度器，长任务用 nohup + tmux；占卡前看 `nvidia-smi`。
