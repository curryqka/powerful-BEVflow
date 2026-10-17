"""训练薄入口。用法：
python scripts/train.py --config configs/exp/R1A_official.yaml --set train.seed=1
"""
import argparse
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
for p in (CODE, UPSTREAM, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bevflow.utils.config import apply_overrides, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="点号覆盖，如 train.seed=1")
    parser.add_argument("--run-dir", default=None, help="缺省为 $EXP/<exp_id>/<run_id>")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    exp_id = cfg["run"].get("exp_id", "R0")
    run_id = cfg["run"].get("run_id") or ""
    assert run_id, "config 必须给 run.run_id（如 20260101_R1A_motionnet_s0）"
    run_dir = args.run_dir or os.path.join(
        os.environ.get("BEVFLOW_EXP", "/media/nas_users/kaneqian/Experiment/bevflow"), exp_id, run_id
    )
    cfg["_code_dir"] = CODE

    from bevflow.trainers.multi_seq import train

    train(cfg, run_dir)
    print("run done:", run_dir)


if __name__ == "__main__":
    main()
