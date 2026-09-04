"""Small CLI for inspecting a released correctness checkpoint."""

from __future__ import annotations
import argparse
import json
import numpy as np
from .checkpoints import load_correctness_ensemble, predict_correctness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--features", required=True)
    args = parser.parse_args()
    features = np.asarray(json.loads(args.features), dtype=np.float32)
    ensemble = load_correctness_ensemble(args.checkpoint)
    per_seed = predict_correctness(ensemble, features, aggregate=False)[:, 0]
    print(json.dumps({
        "mean_correctness": float(per_seed.mean()),
        "per_seed": {
            str(member.seed): float(value)
            for member, value in zip(ensemble.members, per_seed)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
