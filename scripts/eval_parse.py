"""解析上游 eval.py 写出的 eval_displacement.txt，生成 metrics.json（motionnet_v1 协议）。

协议：每档 EPE 的 mean/median 取 5 个评测帧（0.2s 间隔，0.2-1.0s）上的均值；
OA=Mean pixel classification accuracy；MCA=mean instance acc。
"""
import argparse
import json
import re

p = argparse.ArgumentParser()
p.add_argument("--eval-file", required=True)
p.add_argument("--out", required=True)
p.add_argument("--run-id", default="")
p.add_argument("--ckpt", default="")
p.add_argument("--split", default="val")
p.add_argument("--n-samples", type=int, default=None)
args = p.parse_args()

text = open(args.eval_file).read()
blocks = re.findall(
    r"-+\nFor cells within speed range \[([\d.]+), ([\d.]+)\]:\n\n(.*?)\n-+\n", text, re.S
)
assert len(blocks) == 3, f"expected 3 speed tiers, got {len(blocks)}"

tier_names = ["static", "slow", "fast"]
epe = {}
for (lo, hi, body), name in zip(blocks, tier_names):
    frames = re.findall(
        r"Frame (\d+):\nThe mean error is ([\d.eE+-]+|None)\nThe 50% error quantile is ([\d.eE+-]+|None)", body
    )
    assert len(frames) > 0, f"no frames parsed in tier {name}"
    means = [float(m) for _, m, _ in frames if m != "None"]
    medians = [float(q) for _, _, q in frames if q != "None"]
    assert means and medians, f"all-None frames in tier {name}"
    epe[name] = {
        "mean": sum(means) / len(means),
        "median": sum(medians) / len(medians),
        "per_frame_mean": means,
        "per_frame_median": medians,
        "range_ms": [float(lo), float(hi)],
    }

def _num_or_none(s):
    return None if s == "nan" else float(s)

oa = _num_or_none(re.search(r"Mean pixel classification accuracy: ([\d.]+|nan)", text).group(1))
mca = _num_or_none(re.search(r"mean instance acc:\s+([\d.]+|nan)", text).group(1))
per_class = {
    cat: _num_or_none(v)
    for cat, v in re.findall(r"mean cat accuracy of (\w+): ([\d.]+|nan)", text)
}

metrics = {
    "run_id": args.run_id,
    "ckpt": args.ckpt,
    "split": args.split,
    "n_samples": args.n_samples,
    "epe": epe,
    "cls": {"OA": oa, "MCA": mca, "per_class": per_class},
    "meta": {"eval_protocol": "motionnet_v1"},
}

# EPE 必须有限；cls 允许 nan（mini 冒烟的 val 只有 2 个场景，可能整类无格子），记录原因
for tier in epe.values():
    assert all(v == v for v in (tier["mean"], tier["median"])), "NaN in epe metrics"

with open(args.out, "w") as f:
    json.dump(metrics, f, indent=1)
print(f"metrics written: {args.out}")
print(json.dumps({k: {"mean": round(v["mean"], 4), "median": round(v["median"], 4)} for k, v in epe.items()}, indent=1))
print(f"OA={oa} MCA={mca}")
