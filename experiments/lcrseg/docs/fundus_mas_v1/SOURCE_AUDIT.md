# Fundus MAS V1 source audit

Status: `PASS_SOURCE_AUDIT` at `2026-09-01T02:24:00Z`. This was a read-only,
zero-model audit completed before registration and before any MAS data access.

- Paper: Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus
  Rohrbach, and Tinne Tuytelaars, “Memory Aware Synapses: Learning what (not)
  to forget,” ECCV 2018. The paper defines per-sample parameter importance as
  the magnitude of the gradient of the learned function, uses the gradient of
  the squared L2 norm for vector outputs, averages across data points,
  accumulates importance across tasks, and applies a weighted quadratic
  penalty. For global MAS on training data it reports `lambda=1` and states
  that lambda was not tuned.
- First-author code: commit
  `c3e6a855cdde588fb74aeb876f84340eb6090ad5`. `MAS.py` defaults to global L2
  MAS and `reg_lambda=1`, documents `lambda=1` for object recognition, offers
  `b1=True` to mimic online per-sample computation, and accumulates omega.
  `MAS_based_Training.py` computes a summed squared-output L2 objective,
  backpropagates once, averages absolute parameter gradients by sample count,
  and adds the exact `lambda * omega * squared parameter displacement`
  penalty through its gradient.

The segmentation binding is therefore fixed prospectively as raw full-map
logits, squared-L2 sum, batch one, arithmetic mean over every current-site
visible training image, cumulative omega, and lambda one without an added
one-half factor. No paper or author implementation supplies a segmentation-
specific spatial normalization; none is introduced.

Authoritative URLs and exact downloaded-byte hashes are recorded in
`source_audit.json`. No external source is copied or executed.
