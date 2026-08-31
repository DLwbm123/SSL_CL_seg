#!/usr/bin/env python3
"""Registered V2 entry point for the shared real-batch engineering check."""
from __future__ import annotations

from check_model_fisher_ewc_real_batch import main


if __name__ == "__main__":
    main(
        registration_path="docs/fundus_model_fisher_ewc_v2/registration.json",
        registration_sha256="cc86b8518de7ad622a41dc20db896310ebdcf59176d4b62ad58b7e6b6db4b670",
        registration_id="FUNDUS_MODEL_FISHER_EWC_V2",
        method_arm="model_fisher_ewc_v2",
    )
