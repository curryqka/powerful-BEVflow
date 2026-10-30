# SPDX-License-Identifier: Apache-2.0
"""对 run 的最后 N 个 epoch ckpt 逐一评测 val，写 per_ckpt.json（含均值与标准差）。

静态指标对 ckpt 敏感，主表报告 last-10 的均值±标准差，而非单点。
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
    p.add_argument("--val-root", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="motionnet", choices=["motionnet", "priormotion", "priormotion_rvpe", "legomotion"])
    args = p.parse_args()

    ckpt_dir = os.path.join(args.run_dir, "ckpt")
    pairs = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"epoch_(\d+)\.pth", f)
        if m:
            pairs.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    pairs.sort()
    pairs = pairs[-args.n:]
    assert pairs, f"no epoch ckpts in {ckpt_dir}"

    device = torch.device("cuda")
    records = []
    for epoch, path in pairs:
        model = load_model(path, device, args.model)
        m = evaluate(model, args.val_root, split="val", out_dir=None,
                     batch_size=args.bs, num_workers=4, device=device)
        rec = {
            "epoch": epoch,
            "ckpt": path,
            "epe": {k: v["mean"] for k, v in m["epe"].items()},
            "OA": m["cls"]["OA"],
            "MCA": m["cls"]["MCA"],
        }
        records.append(rec)
        print(f"epoch {epoch}: static={rec['epe']['static']:.4f} slow={rec['epe']['slow']:.4f} "
              f"fast={rec['epe']['fast']:.4f} OA={rec['OA']:.4f}")

    def agg(key_path):
        vals = []
        for r in records:
            v = r
            for k in key_path.split("."):
                v = v[k]
            if v is not None:
                vals.append(v)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    summary = {
        "n_ckpts": len(records),
        "epochs": [r["epoch"] for r in records],
        "mean_std": {
            "epe_static": agg("epe.static"), "epe_slow": agg("epe.slow"), "epe_fast": agg("epe.fast"),
            "OA": agg("OA"), "MCA": agg("MCA"),
        },
        "per_ckpt": records,
    }
    out = args.out or os.path.join(args.run_dir, "per_ckpt.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print("written:", out)
    print("mean±std:", json.dumps(summary["mean_std"], indent=1))


if __name__ == "__main__":
    main()
