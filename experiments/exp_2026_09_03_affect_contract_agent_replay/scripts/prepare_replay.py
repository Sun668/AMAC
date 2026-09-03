#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ORDERS = ("TAV", "TVA", "ATV", "AVT", "VTA", "VAT")
STATES = {-1: "negative", 0: "neutral", 1: "positive"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def subset(value):
    return "".join(modality for modality in "TAV" if modality in value)


def state(value):
    if value < -0.1:
        return "negative"
    if value <= 0.1:
        return "neutral"
    return "positive"


def main():
    root = Path(__file__).resolve().parents[3]
    experiment = Path(__file__).resolve().parents[1]
    source = root / "experiments/exp_2026_09_03_amac_chsimsv2_test/results/chsimsv2_test_v1_recovery/predictions.npz"
    baseline = root / "experiments/exp_2026_09_03_amac_competitive_baselines/results/competitive_baselines_v1"
    metrics = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    if metrics["decision"]["selected_estimator"] != "LR":
        raise RuntimeError("竞争性基线未选择 LR，禁止生成当前回放输入")
    parameters = metrics["selected_parameters"]["1"]["LR"]
    predictions = np.load(source, allow_pickle=False)
    scores = np.load(baseline / "scores.npz", allow_pickle=False)["test_lr"]
    ids = predictions["test_ids"].astype(str)
    rows = pd.read_csv(baseline / "per_path.csv")
    rows = rows[(rows["seed"] == 1) & (rows["condition"] == "LR")]
    expected = {(row.clip_id, row.path): json.loads(row.commits) for row in rows.itertuples()}
    destination = experiment / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    replay = destination / "replay.jsonl"
    if replay.exists():
        raise RuntimeError("回放输入已存在，禁止覆盖")
    count = 0
    with replay.open("w", encoding="utf-8") as handle:
        for clip_index, clip_id in enumerate(ids):
            for order_index, order in enumerate(ORDERS):
                observations = []
                for stage in range(3):
                    name = subset(order[:stage + 1])
                    probability = float(scores[(clip_index * len(ORDERS) + order_index) * 2 + stage]) if stage < 2 else 1.0
                    observations.append({
                        "modality": order[stage],
                        "provisional_state": state(float(predictions[f"test_{name}"][clip_index])),
                        "correctness_probability": probability,
                    })
                commits = [None if value is None else STATES[int(value)] for value in expected[(clip_id, order)]]
                record = {
                    "session_id": f"{clip_index}:{order}",
                    "clip_id": clip_id,
                    "path": order,
                    "threshold": parameters["threshold"],
                    "margin": parameters["margin"],
                    "observations": observations,
                    "expected_commits": commits,
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    if count != len(ids) * len(ORDERS):
        raise RuntimeError("回放路径数量错误")
    manifest = {
        "schema": "affect-contract-replay-input-v1",
        "paths": count,
        "observations": count * 3,
        "selected_estimator": "LR",
        "threshold": parameters["threshold"],
        "margin": parameters["margin"],
        "source_hashes": {
            "predictions.npz": sha256(source),
            "baseline_metrics.json": sha256(baseline / "metrics.json"),
            "baseline_per_path.csv": sha256(baseline / "per_path.csv"),
            "baseline_scores.npz": sha256(baseline / "scores.npz"),
        },
        "replay_sha256": sha256(replay),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"code": 0, "message": "Agent 回放输入已生成", "paths": count, "sha256": manifest["replay_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
