"""评测薄入口。用法：
python scripts/eval.py --ckpt <run>/ckpt/last.pt --data-root <val 或 test 目录> \
    --split val --out-dir <结果目录> --run-id <id>
"""
import argparse
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
for p in (CODE, UPSTREAM, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from bevflow.models.loader import load_motionnet  # noqa: E402


def load_model(ckpt_path, device):
    return load_motionnet(ckpt_path, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bs", type=int, default=1, help="逐样本均值严格成立的官方用法是 1")
    parser.add_argument("--worker", type=int, default=2)
    parser.add_argument("--model", default="motionnet", choices=["motionnet", "priormotion"])
    args = parser.parse_args()

    from bevflow.evaluators.motion_eval import evaluate
    from scripts.registry import append as registry_append

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device, args.model)
    metrics = evaluate(
        model, args.data_root, split=args.split, out_dir=args.out_dir,
        batch_size=args.bs, num_workers=args.worker, device=device,
        run_id=args.run_id, ckpt_path=args.ckpt,
    )
    registry_append({
        "run_id": args.run_id or f"eval_{os.path.basename(args.ckpt)}",
        "exp_id": "EVAL", "seed": None, "status": "done",
        "ckpt": args.ckpt, "results_path": args.out_dir,
        "epe": {k: v["mean"] for k, v in metrics["epe"].items()},
        "note": f"split={args.split} bs={args.bs} OA={metrics['cls']['OA']}",
    })
    print("OA:", metrics["cls"]["OA"], "MCA:", metrics["cls"]["MCA"])
    for tier, v in metrics["epe"].items():
        print(f"epe {tier}: mean={v['mean']}")


if __name__ == "__main__":
    main()
