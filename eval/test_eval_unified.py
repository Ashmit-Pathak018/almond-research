import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eval" / "eval_unified.py"

spec = importlib.util.spec_from_file_location("eval_unified", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resolve_dataset_path_from_repo_root():
    resolved = module.resolve_dataset_path("data/longmemeval_oracle.json")
    assert resolved == ROOT / "data" / "longmemeval_oracle.json"
