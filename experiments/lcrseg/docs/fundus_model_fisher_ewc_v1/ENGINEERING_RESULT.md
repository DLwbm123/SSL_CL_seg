# Fundus Model-Fisher EWC V1 engineering result

Status: `FAIL_ENGINEERING_CLOSED`.

The single registered real visible-training admission exited 1 before Fisher
estimation. Parameters created through the shared CUDA runner were on
`cuda:0`, while the registered implementation compared them with the
unindexed device `cuda`. PyTorch treats those device objects as unequal, so
the strict device gate raised `ValueError: model-Fisher device differs from
current model`. The shared formal runner uses the same unindexed device and
would encounter the same failure.

The attempt made 18 training-path model calls and two optimizer updates. It
made zero Fisher model calls and zero `autograd.grad` calls. It stayed within
all prospective budgets. No validation or test role, hidden training label,
formal run, or test evaluation was used.

After the child exited, a zero-model audit rehashed all 1,119 allowed non-test
input files (157,158,028 bytes), the frozen manifests, splits, and checksum
file. All bytes remained unchanged, the child was absent, and GPU7 had
returned to its idle state.

V1 is closed and cannot be retried with changed code under this registration.
Any corrected implementation requires a separately named prospective
registration, a new create-only NAS root, and public source verification
before another real execution.
