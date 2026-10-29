#!/bin/bash
# 发布前私有信息扫描（P7）。只报告，不修改任何文件。
# 用法：bash scripts/audit_release.sh
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)
cd "$CODE"
WARN=0

echo "== 1. 绝对路径 / 用户名 / 内部主机名 =="
if grep -rnE "/home/kaneqian|/media/nas_users|dsw-[0-9]|@mails\.tsinghua" \
    --include="*.py" --include="*.yaml" --include="*.md" --include="*.sh" \
    --exclude-dir=third_party --exclude-dir=.git --exclude-dir=ref \
    . | grep -v "docs/setup/" | head -20; then
    echo "  [WARN] 命中上述模式（docs/setup/ 内的环境文档属内部资料，开源时整体排除）"
    WARN=1
else
    echo "  ok"
fi

echo "== 2. 疑似 token / 密钥 =="
if grep -rnE "(ghp_|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE KEY)" \
    --exclude-dir=.git --exclude-dir=third_party . | head -5; then
    echo "  [WARN] 发现疑似凭证"
    WARN=1
else
    echo "  ok"
fi

echo "== 3. 超过 1MB 的受控文件 =="
find . -type f -size +1M -not -path "./.git/*" -not -path "./third_party/*" \
     -not -path "./data/*" -not -path "./experiment/*" -not -path "./results/*" \
     -not -path "./ref/*" | head -10
echo "  （空即 ok）"

echo "== 4. 私有目录是否会被发布 =="
for d in ref third_party data experiment results docs/setup claude_prompt; do
    if [ -d "$d" ]; then echo "  $d/ 存在——确认发布时排除（ref/third_party/data/experiment/results 已 gitignore；docs/setup 与 claude_prompt 需打包时排除）"; fi
done

exit $WARN
