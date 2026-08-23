"""Guards the engine seam (spec §3.1).

Every module below must be importable and testable without torch installed.
If one of them grows a top-level `import torch`, local development stops
working on a machine without a GPU and this test says so immediately.
"""

import ast
from pathlib import Path

import pytest

GPU_FREE_MODULES = [
    "app/schema.py",
    "app/config.py",
    "app/imaging.py",
    "app/parsing.py",
    "app/validate.py",
    "app/boxes.py",
    "app/extract.py",
]

FORBIDDEN = {"torch", "transformers", "accelerate", "qwen_vl_utils"}


@pytest.mark.parametrize("path", GPU_FREE_MODULES)
def test_module_has_no_top_level_gpu_import(path):
    file = Path(path)
    if not file.exists():
        pytest.skip(f"{path} not implemented yet")
    tree = ast.parse(file.read_text(encoding="utf-8"))
    for node in tree.body:  # top level only; lazy imports inside functions are fine
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (names & FORBIDDEN), f"{path} imports {names & FORBIDDEN} at module level"


def test_qwen_engine_import_does_not_pull_in_torch():
    """app.model must be importable without torch: QwenEngine imports it lazily."""
    tree = ast.parse(Path("app/model.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            assert (mod or "").split(".")[0] not in FORBIDDEN
