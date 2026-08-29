from pathlib import Path

import lcrseg


def test_v04_hidden_gt_code_is_not_imported_by_training_modules() -> None:
    package = Path(lcrseg.__file__).resolve().parent
    training_paths = [
        package / "data",
        package / "engine",
        package / "methods",
        package / "models",
    ]
    forbidden = ("analysis.v0_4", "export_v0_4_diagnostic_features", "diagnostic_records")
    for root in training_paths:
        for path in root.rglob("*.py"):
            if path.name.startswith("._"):
                continue
            source = path.read_text(encoding="utf-8")
            assert not any(token in source for token in forbidden), path


def test_v04_exporter_declares_posthoc_hidden_gt_scope() -> None:
    package = Path(lcrseg.__file__).resolve().parent
    script = package.parent / "scripts" / "export_v0_4_diagnostic_features.py"
    source = script.read_text(encoding="utf-8")
    assert '"hidden_gt_usage": "post_hoc_only"' in source
    assert '"training_imports_this_script": False' in source
