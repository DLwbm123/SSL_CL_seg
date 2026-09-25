# Source provenance

| Input | Binding | Use |
|---|---|---|
| Local 16-page QPrompt-R1 PDF | SHA256 `b4944a31a8bb1e6911de00f62f4d39f501daedfd5ce820b788822e92539f362b` | User confirmed this copy as R0 source. The attached plan quoted a different SHA (`96b964dd…`); the discrepancy is retained. PDF is private and is not in this code tree. |
| [QPrompt-R1 official repository](https://github.com/straybird2333/QPrompt-R1) | README-only on 2026-09-25 | Attribution only; no official implementation imported. |
| [DINOv2 official source](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) | commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, per-file hashes in `third_party/DINOV2_SOURCE_SHA256.json` | Minimal ViT-S/14 import closure under `third_party/`; Apache-2.0 `LICENSE` retained. Normal DINOv2 code/weights are Apache-2.0 per upstream README. |
| [Official ViT-S/14 weight](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | **PENDING** local single-file acquisition and SHA256 binding | Must be stored on confirmed new remote-home. No implicit `torch.hub` download. |
| Existing U-Net body | `experiments/lcrseg/di_dmpa_jascl/modeling.py` at base `f6b96973632b9fa8a2c3578e7fa7a18c65ff47ca` | 16/32/64/128 GroupNorm encoder/decoder geometry, ported into `qprompt/models.py`. |
| Canonical data definitions | `experiments/lcrseg/single_teacher_scd_v0_1/data.py` at same base | RGB/label shape, hashes, role restrictions; minimal independent reader in `qprompt/data.py`. |
| Attached migration helper | ZIP dated 2026-09-24 | `migration/minimal_manifest.py` and fixture tests copied, then adapted for separate metadata/payload roots; original frozen metadata unchanged. |

The selected pretrained ViT is smaller than the paper's principal ViT-L setting. U-Net starts from scratch. Cross-backbone absolute-score comparisons cannot isolate architecture from pretraining.
