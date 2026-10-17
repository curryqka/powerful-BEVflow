"""路径唯一来源：环境变量优先，缺省值取 configs/paths.yaml 同款常量。"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    "BEVFLOW_CODE": PROJECT_ROOT,
    "BEVFLOW_EXP": "/media/nas_users/kaneqian/Experiment/bevflow",
    "BEVFLOW_RESULTS": "/media/nas_users/kaneqian/results/bevflow",
    "BEVFLOW_DATA": "/media/nas_users/kaneqian/data/bevflow",
}


def root(name):
    assert name in DEFAULTS, f"unknown root: {name}"
    return os.environ.get(name, DEFAULTS[name])


def code():
    return root("BEVFLOW_CODE")


def exp():
    return root("BEVFLOW_EXP")


def results():
    return root("BEVFLOW_RESULTS")


def data():
    return root("BEVFLOW_DATA")


def assert_under(path, root_dir):
    real = os.path.realpath(path)
    allowed = os.path.realpath(root_dir)
    assert real == allowed or real.startswith(allowed + os.sep), (
        f"path escapes {root_dir}: {path} -> {real}"
    )


def writable_path(path):
    """写文件入口用：断言目标在这四个根之下。"""
    target = os.path.dirname(os.path.abspath(path)) or path
    for r in (code(), exp(), results(), data()):
        try:
            assert_under(target, r)
            return path
        except AssertionError:
            continue
    raise AssertionError(f"write target not under any project root: {path}")
