from __future__ import annotations

import argparse
from pathlib import Path

from lcrseg.common import DATA_ROOT
from lcrseg.preprocess import PreprocessOptions


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--preprocess-version", default="v1")
    parser.add_argument("--execute", action="store_true", help="Write derived HDF5 after all gates pass.")


def options_from_args(args: argparse.Namespace) -> PreprocessOptions:
    return PreprocessOptions(
        output_root=args.output_root,
        preprocess_version=args.preprocess_version,
        execute=bool(args.execute),
    )
