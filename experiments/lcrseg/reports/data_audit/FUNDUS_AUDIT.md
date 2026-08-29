# Fundus audit

- Total paired records: `660`
- Per-site audit: `{"Drishti_GS": {"candidate_larger_foreground_value": 255, "candidate_smaller_foreground_value": 128, "cases": 101, "mask_value_profiles": {"[0, 128, 255]": 101}, "mean_center_crop_retention": 1.0, "minimum_center_crop_retention": 1.0, "qc_overlays_written": 20}, "REFUGE": {"candidate_larger_foreground_value": 255, "candidate_smaller_foreground_value": 128, "cases": 400, "mask_value_profiles": {"[0, 128, 255]": 400}, "mean_center_crop_retention": 1.0, "minimum_center_crop_retention": 1.0, "qc_overlays_written": 20}, "RIM_ONE_r3": {"candidate_larger_foreground_value": 255, "candidate_smaller_foreground_value": 128, "cases": 159, "mask_value_profiles": {"[0, 128, 255]": 159}, "mean_center_crop_retention": 1.0, "minimum_center_crop_retention": 1.0, "qc_overlays_written": 20}}`

The center crop is assessed using hidden GT only for offline QC. The grayscale code mapping remains intentionally unconfirmed.
