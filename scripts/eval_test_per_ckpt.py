"""对 run 的最后 N 个 epoch ckpt 在 test 集上逐一评测（bs1），写 per_ckpt_test.json。

与 eval_per_ckpt（val）互补：验证指标对 ckpt 的敏感性，并支持按 val 选 ckpt、test 报数。
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
sys.path.insert(0, CODE)
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk"))

from bevflow.evaluators.motion_eval import evaluate  # noqa: E402
from bevflow.models.loader import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--model", default="priormotion")
    args = p.parse_args()

    ckpt_dir = os.path.join(args.run_dir, "ckpt")
    pairs = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"epoch_(\d+)\.pth", f)
        if m:
            pairs.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    pairs.sort()
    pairs = pairs[-args.n:]
    assert pairs

    device = torch.device("cuda")
    records = []
    for epoch, path in pairs:
        model = load_model(path, device, args.model)
        m = evaluate(model, args.test_root, split="test", out_dir=None,
                     batch_size=1, num_workers=4, device=device)
        rec = {"epoch": epoch,
               "epe": {k: v["mean"] for k, v in m["epe"].items()},
               "fast_at_1s": m["epe"]["fast"]["per_frame_mean"][-1],
               "slow_at_1s": m["epe"]["slow"]["per_frame_mean"][-1],
               "OA": m["cls"]["OA"], "MCA": m["cls"]["MCA"]}
        records.append(rec)
        print(f"epoch {epoch}: fast@1s={rec['fast_at_1s']:.4f} slow@1s={rec['slow_at_1s']:.4f} "
              f"static={rec['epe']['static']:.4f} MCA={rec['MCA']:.4f}", flush=True)

    def agg(key):
        vals = [r[key] for r in records if r[key] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out = {"n_ckpts": len(records), "epochs": [r["epoch"] for r in records],
           "mean_std": {k: agg(k) for k in ("fast_at_1s", "slow_at_1s", "OA", "MCA")},
           "per_ckpt": records}
    path = os.path.join(args.run_dir, "per_ckpt_test.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("written:", path)


if __name__ == "__main__":
    main()
