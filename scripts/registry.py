# SPDX-License-Identifier: Apache-2.0
"""结果登记：$RESULTS/registry.jsonl 是唯一真源，只追加不改旧行，同一 run_id 以最后一行为准。

用法：
  python scripts/registry.py --run-id X --exp-id R1A --seed 0 --status running \
      --exp-path ... --results-path ... [--epe-static .. --epe-slow .. --epe-fast ..] [--note ..]
  python scripts/registry.py --md   # 生成 docs/experiments/registry.md
"""
import argparse
import datetime
import json
import os
import subprocess

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("BEVFLOW_RESULTS", "/media/nas_users/kaneqian/results/bevflow")
REGISTRY = os.path.join(RESULTS, "registry.jsonl")


def append(record):
    os.makedirs(RESULTS, exist_ok=True)
    record["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    record["git"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True
    ).stdout.strip()
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("registry:", json.dumps(record, ensure_ascii=False))


def load_all():
    if not os.path.exists(REGISTRY):
        return []
    with open(REGISTRY) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_by_run(records):
    latest = {}
    for r in records:
        latest[r["run_id"]] = r
    return latest


def write_md():
    latest = latest_by_run(load_all())
    rows = ["| run_id | exp_id | seed | status | epe static/slow/fast | exp 路径 |",
            "|---|---|---|---|---|---|"]
    for r in sorted(latest.values(), key=lambda x: x.get("ts", "")):
        epe = r.get("epe") or {}
        epe_str = "/".join(str(epe.get(k, "-")) for k in ("static", "slow", "fast"))
        rows.append(f"| {r['run_id']} | {r.get('exp_id','-')} | {r.get('seed','-')} "
                    f"| {r['status']} | {epe_str} | {r.get('exp_path','-')} |")
    out = os.path.join(CODE, "docs", "experiments", "registry.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("# 实验登记（自动生成，不手改）\n\n由 scripts/registry.py --md 生成。\n\n")
        f.write("\n".join(rows) + "\n")
    print("written:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id")
    p.add_argument("--exp-id")
    p.add_argument("--seed", type=int)
    p.add_argument("--status", choices=["running", "done", "failed"])
    p.add_argument("--config", default="")
    p.add_argument("--exp-path", default="")
    p.add_argument("--results-path", default="")
    p.add_argument("--epe-static", type=float)
    p.add_argument("--epe-slow", type=float)
    p.add_argument("--epe-fast", type=float)
    p.add_argument("--note", default="")
    p.add_argument("--md", action="store_true", help="生成 markdown 视图")
    args = p.parse_args()

    if args.md:
        write_md()
        return
    assert args.run_id and args.status, "--run-id 与 --status 必填"
    epe = {k: v for k, v in (("static", args.epe_static), ("slow", args.epe_slow), ("fast", args.epe_fast))
           if v is not None}
    append({
        "run_id": args.run_id, "exp_id": args.exp_id, "seed": args.seed,
        "status": args.status, "config": args.config, "epe": epe,
        "exp_path": args.exp_path, "results_path": args.results_path, "note": args.note,
    })


if __name__ == "__main__":
    main()
