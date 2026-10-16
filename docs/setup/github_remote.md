# GitHub 推送配置（容器内）

从容器把代码推到自己的 GitHub。本机已配好，本文记录全过程，供下次重建或换机器参考。配置时间：2026-10-16。

## 一、生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "kaneqian@dsw"
```

`.pub` 结尾的是公钥，可以公开。私钥 `~/.ssh/id_ed25519` 不外传，也不进对话。

## 二、公钥加到 GitHub

网页 https://github.com/settings/ssh/new 。Key type 选 Authentication Key，粘贴 `.pub` 整行内容，保存。

验证：

```bash
ssh -T git@github.com
# 看到 "Hi <用户名>! You've successfully authenticated" 即成功
```

## 三、首次连接加 host key

```bash
ssh-keyscan github.com >> ~/.ssh/known_hosts
ssh-keygen -lf ~/.ssh/known_hosts | grep github
```

对照 GitHub 官方指纹，ed25519 是 `SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`。对不上就停，别连。

## 四、配 remote 与推送

```bash
git remote add origin git@github.com:curryqka/powerful-BEVflow.git
git push -u origin bevflow    # -u 建立跟踪，以后直接 git push
```

## 五、分支策略（本仓库）

`powerful-BEVflow` 是总仓，现有分支按项目命名：`PriorMotion`，`MotionNet`，`hesai`，`main`，`qka-G`。BEVflow 新工作在 `bevflow` 分支，历史与旧分支无关（独立起点，P1 根提交）。

原则：新工作开新分支，命名跟项目走。不重排别人的分支。清理旧分支前，先列清单（分支名，最后提交时间），owner 确认后再动。

## 六、本机的坑

- git 默认凭据助手指向不存在的 vscode socket（报 `ECONNREFUSED`），HTTPS 认证不可用。直接用 SSH。
- `git init` 默认分支名是 `master`。推之前 `git branch -m master bevflow` 改名。
- `gh` CLI 未安装。需要时从 github releases 下载二进制放 `~/.local/bin`。
- 私有仓库走 API 返回 404，不代表仓库不存在。判断仓库可连性用 `git ls-remote`。

## 七、自动化约定

每个阶段检查点：commit 后直接 `git push`。`bevflow` 分支已跟踪 `origin/bevflow`。
