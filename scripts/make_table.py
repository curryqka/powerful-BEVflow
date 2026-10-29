# SPDX-License-Identifier: Apache-2.0
"""结果表生成：输入 run_id 列表，从 registry 找到 metrics.json，汇总 markdown 表。

用法：python scripts/make_table.py --runs R1B_s0_ep45_test ...
每个数字都能顺着 run_id → registry 的 results_path → metrics.json 追溯。
"""
import argparse
import json
import os
import subprocess

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("BEVFLOW_RESULTS", "/media/nas_users/kaneqian/results/bevflow")

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True, help="registry 里的 run_id")
p.add_argument("--last10", action="store_true", help="改读 run 目录的 per_ckpt.json（last-10 均值±标准差）")
args = p.parse_args()

records = {}
with open(os.path.join(RESULTS, "registry.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        records[r["run_id"]] = r

rows = []
missing = []
for rid in args.runs:
    r = records.get(rid)
    if r is None:
        missing.append(rid)
        continue
    if args.last10:
        pc = os.path.join(r.get("exp_path", ""), "per_ckpt.json")
        if not os.path.exists(pc):
            missing.append(rid + "(no per_ckpt.json)")
            continue
        d = json.load(open(pc))["mean_std"]
        def fmt(k):
            v = d.get(k)
            return f"{v['mean']:.4f}±{v['std']:.4f}" if v else "-"
        rows.append((rid, "val(last-10)", fmt("epe_static"), fmt("epe_slow"), fmt("epe_fast"), fmt("OA"), fmt("MCA")))
    else:
        mj = os.path.join(r.get("results_path", ""), "metrics.json")
        if not os.path.exists(mj):
            missing.append(rid + "(no metrics.json)")
            continue
        d = json.load(open(mj))
        rows.append((
            rid, d.get("split", "-"),
            f"{d['epe']['static']['mean']:.4f}", f"{d['epe']['slow']['mean']:.4f}", f"{d['epe']['fast']['mean']:.4f}",
            f"{d['cls']['OA']:.4f}", f"{d['cls']['MCA']:.4f}" if d["cls"]["MCA"] is not None else "-",
        ))

header_rows = [
    "| run_id | split | epe static | epe slow | epe fast | OA | MCA |",
    "|---|---|---|---|---|---|---|",
]
body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
mode = "per_ckpt.json (val last-10)" if args.last10 else "metrics.json"
print(f"来源: {mode} | 生成: make_table.py | git {git}")
print()
print("\n".join(header_rows + body))
if missing:
    import sys

    print(f"\n[缺数据: {missing}]", file=sys.stderr)
