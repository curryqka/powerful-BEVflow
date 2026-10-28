"""扩展口径评测 CLI（D8）：空间分桶/x-y-径向-切向分解/实例稳定性。"""
import argparse
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
for p in (CODE, UPSTREAM, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from bevflow.models.loader import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--model", default="motionnet", choices=["motionnet", "priormotion", "priormotion_rvpe"])
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--worker", type=int, default=4)
    args = p.parse_args()

    from bevflow.evaluators.motion_eval_general import evaluate_general

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device, args.model)
    m = evaluate_general(model, args.data_root, split=args.split, out_dir=args.out_dir,
                         batch_size=args.bs, num_workers=args.worker, device=device,
                         run_id=args.run_id, ckpt_path=args.ckpt)
    st = m["stability"]
    print(f"stability: pred_var={st['mean_pred_speed_variance']} gt_var={st['mean_gt_speed_variance']} "
          f"instances={st['n_instances']}")
    print("spatial bins written to", os.path.join(args.out_dir, "metrics_general.json"))


if __name__ == "__main__":
    main()
