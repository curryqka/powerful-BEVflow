"""golden 对拍：预处理。

三层比较：
1. 新 preprocess（跑两遍，不同进程）必须逐位一致——确定性是本移植相对上游的改进，
   上游用 set 迭代重排实例，顺序随哈希种子变化，重跑自身即有厘米级差异（已实验证实）。
2. 与上游参照（冒烟时 gen_data 生成的 116_*）比较：除 pixel_instance_ids（重标签等价）与
   disp_field（重叠同类别框像素，允许极小比例厘米级差异）外逐位相等。
3. valid_pixel_map 等结构字段必须逐位相等。
"""
import os
import subprocess
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NUSCENES = "/media/nas_users/kaneqian/data/Nuscences"
REF_ROOT = "/media/nas_users/kaneqian/data/bevflow/preprocessed/mini/train"


def _generate(out_dir):
    upstream = os.path.join(CODE, "third_party", "MotionNet")
    code = (
        "import sys;"
        f"sys.path.insert(0, {CODE!r});"
        f"sys.path.insert(0, {upstream!r});"
        f"sys.path.insert(0, {os.path.join(upstream, 'nuscenes-devkit', 'python-sdk')!r});"
        "from bevflow.preprocess.generate import generate_scenes, load_nuscenes;"
        f"nusc = load_nuscenes({RAW_NUSCENES!r});"
        f"print(generate_scenes(nusc, [116], 'train', {out_dir!r}, log_every=1000))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, f"generate failed: {r.stdout[-2000:]}\n{r.stderr[-2000:]}"


def _is_pure_relabel(a, b):
    if len(set(a)) != len(set(b)):
        return False
    mapping = {}
    for x, y in zip(a, b):
        if x in mapping and mapping[x] != y:
            return False
        mapping[x] = y
    return len(set(mapping.values())) == len(set(a))


def _compare_seq_dir(new_root, ref_root, seq, ref_files):
    for f in ref_files:
        ref = np.load(os.path.join(ref_root, seq, f), allow_pickle=True).item()
        new = np.load(os.path.join(new_root, seq, f), allow_pickle=True).item()
        assert set(ref.keys()) == set(new.keys()), f"keys mismatch in {seq}/{f}"
        for k in ref:
            rv, nv = ref[k], new[k]
            if not isinstance(rv, np.ndarray):
                assert rv == nv, f"scalar mismatch {k} in {seq}/{f}"
                continue
            assert rv.dtype == nv.dtype, f"dtype mismatch {k} in {seq}/{f}"
            if np.array_equal(rv, nv):
                continue
            if k == "pixel_instance_ids":
                # 重叠同类别框的像素归属依赖实例处理顺序（上游 set 迭代序，本移植为排序序）。
                # 实测上游重跑自身：前景结构始终一致；实例个数可在 4/34 序列差 1；
                # id 重排比例可高达 96%。故判据：前景结构逐位一致 + 实例个数差 ≤2。
                if not _is_pure_relabel(rv, nv):
                    assert np.array_equal(rv > 0, nv > 0), f"foreground pattern mismatch in {seq}/{f}"
                    assert abs(len(set(rv)) - len(set(nv))) <= 2, (
                        f"instance count differs by >2 in {seq}/{f}: {len(set(rv))} vs {len(set(nv))}"
                    )
            elif k == "disp_field":
                d = np.linalg.norm(rv - nv, axis=-1)
                frac = np.sum(d > 1e-3) / d.size
                assert frac < 5e-4, f"too many disp pixels differ ({frac:.2%}) in {seq}/{f}"
                assert np.nanmax(d) < 0.5, f"disp diff too large ({np.nanmax(d):.3f} m) in {seq}/{f}"
                assert np.array_equal(np.isnan(rv), np.isnan(nv)), f"nan pattern mismatch in {seq}/{f}"
            else:
                raise AssertionError(f"value mismatch {k} in {seq}/{f}")


def test_preprocess_scene116_deterministic_and_matches_upstream(tmp_path):
    out1 = str(tmp_path / "run1")
    out2 = str(tmp_path / "run2")
    _generate(out1)
    _generate(out2)

    ref_seqs = sorted(d for d in os.listdir(REF_ROOT) if d.startswith("116_"))
    new_seqs = sorted(d for d in os.listdir(out1))
    assert new_seqs == ref_seqs, f"seq dir mismatch: {len(new_seqs)} vs {len(ref_seqs)}"

    # 1. 自身确定性：两进程产物逐位一致
    for seq in ref_seqs:
        for f in sorted(os.listdir(os.path.join(REF_ROOT, seq))):
            a = np.load(os.path.join(out1, seq, f), allow_pickle=True).item()
            b = np.load(os.path.join(out2, seq, f), allow_pickle=True).item()
            for k in a:
                av, bv = a[k], b[k]
                if isinstance(av, np.ndarray):
                    assert np.array_equal(av, bv), f"nondeterministic {k} in {seq}/{f}"
                else:
                    assert av == bv, f"nondeterministic scalar {k} in {seq}/{f}"

    # 2/3. 与上游参照比较（容差见文件头注释）
    for seq in ref_seqs:
        _compare_seq_dir(out1, REF_ROOT, seq, sorted(os.listdir(os.path.join(REF_ROOT, seq))))
