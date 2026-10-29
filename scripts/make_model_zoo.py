"""从 registry.jsonl 生成 docs/model_zoo.md（P7，不手写）。"""
import json
import os
import subprocess

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("BEVFLOW_RESULTS", "/media/nas_users/kaneqian/results/bevflow")
EXP = os.environ.get("BEVFLOW_EXP", "/media/nas_users/kaneqian/Experiment/bevflow")

ZOO_RUNS = [
    # (run_id, 模型名, 说明)
    ("official_test_bs1", "MotionNet（官方权重）", "Zenodo 8215963，test"),
    ("PM_DSpG_s0_ep45_test", "PriorMotion DSpG s0", "Table 7(e) 对照臂"),
    ("PM_DSpG_s1_ep45_test", "PriorMotion DSpG s1", "Table 7(e) 对照臂"),
    ("PM_rvpe_s0_test", "PriorMotion RVpE s0", "全配置，Table 2 对照"),
    ("PM_rvpe_s1_test", "PriorMotion RVpE s1", "全配置，Table 2 对照"),
]


def main():
    latest = {}
    with open(os.path.join(RESULTS, "registry.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            latest[r["run_id"]] = r

    def metrics_for(rid, record):
        rp = record.get("results_path") or ""
        cand = [os.path.join(rp, "metrics.json"),
                os.path.join(rp, "metrics_general.json")]
        for c in cand:
            if os.path.exists(c):
                return json.load(open(c))
        return None

    rows = ["| run_id | 模型 | epe static | slow@1.0s | fast@1.0s | OA | MCA | ckpt | 说明 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for rid, model_name, note in ZOO_RUNS:
        rec = latest.get(rid)
        if rec is None:
            rows.append(f"| {rid} | {model_name} | - | - | - | - | - | - | {note} |")
            continue
        m = metrics_for(rid, rec)
        if m:
            e = m["epe"]
            rows.append(
                f"| {rid} | {model_name} | {e['static']['mean']:.4f} | "
                f"{e['slow']['per_frame_mean'][-1]:.4f} | {e['fast']['per_frame_mean'][-1]:.4f} | "
                f"{m['cls']['OA']:.4f} | {m['cls']['MCA']:.4f} | {os.path.basename(rec.get('ckpt','-'))} | {note} |")
        else:
            rows.append(f"| {rid} | {model_name} | 无 metrics | | | | | | {note} |")

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    out = os.path.join(CODE, "docs", "model_zoo.md")
    with open(out, "w") as f:
        f.write("# Model Zoo（自动生成，不手改）\n\n由 scripts/make_model_zoo.py 生成，git " + git + "\n\n")
        f.write("\n".join(rows) + "\n")
    print("written:", out)


if __name__ == "__main__":
    main()
