"""数值检查可视化 PNG：存在且 >5KB，尺寸达标，像素 std>0，误差图非全零（预测不等于 GT）。"""
import argparse
import json
import os

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
args = p.parse_args()

m = json.load(open(args.manifest))
base = os.path.dirname(os.path.abspath(args.manifest))
checks = []
for s in m["samples"]:
    for role in ("quiver", "err", "staticerr"):
        f = os.path.join(base, s[role])
        ok_file = os.path.exists(f) and os.path.getsize(f) > 5120
        assert ok_file, f"missing or too small: {f}"
        img = np.asarray(Image.open(f).convert("RGB"))
        assert img.shape[0] >= 400 and img.shape[1] >= 400, f"dims too small: {f} {img.shape}"
        assert img.std() > 0, f"constant image: {f}"
        checks.append({"file": s[role], "std": round(float(img.std()), 1)})

    err_img = np.asarray(Image.open(os.path.join(base, s["err"])).convert("RGB"))
    assert err_img.std() > 2, f"error map near-constant, pred may equal gt: idx {s['idx']}"

out = {"n_checks": len(checks), "samples": len(m["samples"]), "status": "PASS", "checks": checks}
with open(os.path.join(base, "viz_check.json"), "w") as f:
    json.dump(out, f, indent=1)
print("VIZ CHECK PASS", len(checks), "files,", len(m["samples"]), "samples")
