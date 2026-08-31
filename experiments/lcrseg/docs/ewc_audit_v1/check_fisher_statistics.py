"""Closed-form source-audit checks: no model, data loader, Torch or optimizer."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads((args.sources / "SOURCES.json").read_text())
    for item in sources["files"]:
        path = args.sources / item["archive_path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "source hash differs")
    legacy = (args.sources / "local/experiments/lcrseg/lcrseg/methods/ewc.py").read_text()
    external = (args.sources / "external/models/cl/continual_learner.py").read_text()
    require('losses["loss_sup"].backward()' in legacy and 'parameter.grad.detach().float().square()' in legacy,
            "reviewed legacy estimator changed")
    require('if index > self.fisher_n:' in external and 'p/index for n, p in est_fisher_info.items()' in external,
            "reviewed external count logic changed")
    # For Bernoulli logit theta=0 and labels 0,1, NLL gradients are +1/2,-1/2.
    gradients = [0.5, -0.5]
    mean_square = sum(g * g for g in gradients) / len(gradients)
    square_mean = (sum(gradients) / len(gradients)) ** 2
    require(mean_square == 0.25 and square_mean == 0, "batch cancellation oracle differs")
    # A second example separates the model expectation from observed-label squares.
    probability = 0.8
    model_fisher = (1 - probability) * probability**2 + probability * (probability - 1)**2
    empirical_fisher = probability**2  # Both observed labels are zero.
    require(math.isclose(model_fisher, 0.16) and math.isclose(empirical_fisher, 0.64), "label expectation oracle differs")
    counts = []
    for length, cap in [(1, None), (2, None), (3, 1), (503, 500)]:
        processed = 0
        for index in range(length):
            if cap is not None and index > cap:
                break
            processed += 1
        counts.append(dict(dataset_length=length, cap=cap, processed=processed, divisor=index))
    require([(r["processed"], r["divisor"]) for r in counts] == [(1, 0), (2, 1), (2, 2), (501, 501)],
            "external loop counterexample differs")
    print(json.dumps(dict(status="PASS_ANALYTIC_SOURCE_CHECK", batch_mean_squared=square_mean,
                          mean_individual_squared=mean_square, model_fisher=model_fisher,
                          empirical_fisher=empirical_fisher, external_count_examples=counts,
                          source_files_verified=len(sources["files"]), real_model_forwards=0,
                          synthetic_network_forwards=0, gradient_calls=0, optimizer_steps=0,
                          dataset_access=False), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
