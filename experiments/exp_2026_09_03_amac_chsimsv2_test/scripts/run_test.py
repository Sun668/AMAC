#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def score_map(metadata, values):
    return {(item[0], item[1], item[2]): float(values[index]) for index, item in enumerate(metadata)}


def aggregate(rows):
    commits = sum(row["prefinal_commits"] for row in rows); errors = sum(row["prefinal_errors"] for row in rows)
    return {"paths": len(rows), "prefinal_commits": commits, "prefinal_errors": errors, "prefinal_committed_error_rate": errors / commits if commits else None, "committed_revision_rate": sum(row["revisions"] for row in rows) / len(rows), "premature_exposure_rate": sum(row["premature"] for row in rows) / len(rows), "stage_two_coverage": sum(row["stage2_covered"] for row in rows) / len(rows), "time_to_first_commit": sum(row["time_to_first"] for row in rows) / len(rows), "final_state_identity": sum(row["final_identity"] for row in rows) / len(rows)}


def evaluate(base, records, condition, parameters, delegate=None):
    rows = []
    for clip_id, order, label, predictions, scores in records:
        row = base.path_row(clip_id, order, delegate or condition, label, predictions, scores, parameters.get("threshold", 0.5), parameters.get("margin", 0.0)); row["condition"] = condition; row.pop("policy", None); rows.append(row)
    return rows


def summarize(per_seed, conditions):
    fields = ["prefinal_committed_error_rate", "committed_revision_rate", "premature_exposure_rate", "stage_two_coverage", "time_to_first_commit"]
    result = {}
    for condition in conditions:
        result[condition] = {}
        for field in fields:
            values = [per_seed[str(seed)]["conditions"][condition][field] for seed in map(int, per_seed)]
            result[condition][field] = {"mean": None, "std": None} if any(value is None for value in values) else {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    started = time.time(); snapshot_path, output = Path(args.parameter_snapshot).resolve(), Path(args.condition_dir).resolve()
    if output.exists(): raise RuntimeError("一次性测试输出目录已存在，禁止覆盖或重跑")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")); snapshot_hash = sha256_file(snapshot_path); root = Path(__file__).resolve().parents[3]
    for key in ("runner", "validator", "development_runner", "matrix_runner"):
        if sha256_file(root / snapshot["environment"][f"{key}_path"]) != snapshot["environment"][f"{key}_sha256"]: raise RuntimeError(f"{key} 哈希与冻结快照不一致")
    for key in ("decision", "metrics", "validator_report"):
        path = root / snapshot["upstream_validation"][f"{key}_path"]
        if sha256_file(path) != snapshot["upstream_validation"][f"{key}_sha256"]: raise RuntimeError(f"上游 {key} 哈希不一致")
    output.mkdir(parents=True)
    matrix = load_module("matrix", root / snapshot["environment"]["matrix_runner_path"]); base = matrix.load_base(root)
    dataset_path = root / snapshot["dataset"]["path"]
    if sha256_file(dataset_path) != snapshot["dataset"]["sha256"]: raise RuntimeError("数据集哈希不一致")
    with dataset_path.open("rb") as handle: dataset = pickle.load(handle)
    train, test = dataset[snapshot["dataset"]["training_split"]], dataset[snapshot["dataset"]["evaluation_split"]]
    train_ids, test_ids = np.asarray(train["id"]).astype(str), np.asarray(test["id"]).astype(str)
    train_labels = np.asarray(train["regression_labels"], dtype=np.float32).reshape(-1); test_labels = np.asarray(test["regression_labels"], dtype=np.float32).reshape(-1)
    if len(train_ids) != snapshot["dataset"]["expected_train"] or len(test_ids) != snapshot["dataset"]["expected_evaluation"]: raise RuntimeError("样本数与冻结快照不一致")
    train_pooled = {"T": base.pool_modality(train["text"]), "A": base.pool_modality(train["audio"]), "V": base.pool_modality(train["vision"])}
    test_pooled = {"T": base.pool_modality(test["text"]), "A": base.pool_modality(test["audio"]), "V": base.pool_modality(test["vision"])}
    del dataset, train, test
    groups = np.asarray([base.source_group(value) for value in train_ids]); cfg = snapshot["models"]["base"]
    train_predictions, test_predictions = base.grouped_ridge_predictions(train_pooled, test_pooled, train_labels, groups, cfg["alpha"], cfg["group_folds"], cfg["solver"])
    official_anchor = {subset: matrix.official_metrics(test_predictions[subset], test_labels) for subset in base.SUBSETS}
    orders = snapshot["sampling"]["arrival_orders"]
    train_features, targets, metadata = base.build_events(train_ids, train_labels, train_predictions, base.modality_quality(train_pooled), orders)
    test_features, _, test_metadata = base.build_events(test_ids, test_labels, test_predictions, base.modality_quality(test_pooled), orders)
    train_features[:, snapshot["preprocessing"]["quality_feature_start"]:] = 0.0; test_features[:, snapshot["preprocessing"]["quality_feature_start"]:] = 0.0
    indices = snapshot["preprocessing"]["history_feature_indices"]; train_features[:, indices] = 0.0; test_features[:, indices] = 0.0
    targets[:, 1] = targets[:, 0]
    config = dict(snapshot["models"]["risk_predictor"]); config.update(snapshot["training"])
    conditions = snapshot["policy"]["conditions"]; all_rows, per_seed, models, risks = [], {}, {}, []
    for seed in snapshot["sampling"]["seeds"]:
        np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
        model, scaler, _, _, training = base.train_amac(train_features, targets, metadata, config, seed)
        _, test_heads = base.predict_scores(model, scaler, test_features); risks.append(test_heads[:, 0])
        records = base.create_records(test_ids, test_labels, test_predictions, orders, score_map(test_metadata, test_heads[:, 0]))
        parameters = snapshot["policy"]["selected_parameters"][str(seed)]
        seed_rows = []
        for condition in ("B0", "B1", "B2", "B3", "B4", "O1"):
            seed_rows.extend(evaluate(base, records, condition, parameters.get(condition, {"threshold": 0.5, "margin": 0.0})))
        seed_rows.extend(evaluate(base, records, "H0", parameters["H0"], "M0"))
        for row in seed_rows: row["seed"] = seed
        all_rows.extend(seed_rows)
        condition_metrics = {condition: aggregate([row for row in seed_rows if row["condition"] == condition]) for condition in conditions}
        comparisons = {}
        for comparator in ("B2", "B3", "B4"):
            own, other = condition_metrics["H0"], condition_metrics[comparator]
            effect = {"error_reduction": other["prefinal_committed_error_rate"] - own["prefinal_committed_error_rate"], "revision_relative_reduction": (other["committed_revision_rate"] - own["committed_revision_rate"]) / other["committed_revision_rate"] if other["committed_revision_rate"] else None}
            effect.update(matrix.paired_intervals(seed_rows, "H0", comparator, snapshot["statistics"]["bootstrap_repetitions"], seed * 1000 + sum(map(ord, "H0" + comparator))))
            comparisons[comparator] = effect
        per_seed[str(seed)] = {"conditions": condition_metrics, "comparisons": comparisons, "parameters": parameters, "training": training}
        models[str(seed)] = {"state_dict": copy.deepcopy(model.state_dict()), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    aggregate_summary = summarize(per_seed, conditions); gates = snapshot["gates"]; seeds = snapshot["sampling"]["seeds"]
    values = lambda comparator, field: [per_seed[str(seed)]["comparisons"][comparator][field] for seed in seeds]
    claim_checks = {"coverage": all(per_seed[str(seed)]["conditions"]["H0"]["stage_two_coverage"] >= gates["coverage_min"] for seed in seeds), "vs_b4_error": float(np.mean(values("B4", "error_reduction"))) >= gates["vs_b4_error_reduction_min"] and all(v > 0 for v in values("B4", "error_reduction")), "vs_b4_revision": float(np.mean(values("B4", "revision_relative_reduction"))) >= gates["vs_b4_revision_relative_min"] and all(v > 0 for v in values("B4", "revision_relative_reduction")), "vs_b4_intervals": all(per_seed[str(seed)]["comparisons"]["B4"]["error_reduction_ci95"][0] > 0 and per_seed[str(seed)]["comparisons"]["B4"]["revision_relative_reduction_ci95"][0] > 0 for seed in seeds), "vs_b2": all(v > 0 for v in values("B2", "error_reduction")) and all(v > 0 for v in values("B2", "revision_relative_reduction")), "b3_context": all(per_seed[str(seed)]["conditions"]["B3"]["stage_two_coverage"] < gates["coverage_min"] or (per_seed[str(seed)]["comparisons"]["B3"]["error_reduction"] > 0 and per_seed[str(seed)]["comparisons"]["B3"]["revision_relative_reduction"] > 0) for seed in seeds)}
    validity_checks = {"five_seeds_complete": len(per_seed) == 5, "final_identity": all(per_seed[str(seed)]["conditions"][condition]["final_state_identity"] == 1.0 for seed in seeds for condition in conditions), "frozen_selected_condition": snapshot["policy"]["selected_condition"] == "H0"}
    common = {"run_id": output.name, "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"]}
    metrics = {"schema": "amac-chsimsv2-test-metrics-v1", "status": "completed", **common, "official_anchor": official_anchor, "per_seed": per_seed, "aggregate": aggregate_summary, "validity_checks": validity_checks, "claim_checks": claim_checks}
    decision = {"schema": "amac-chsimsv2-test-decision-v1", "status": "completed", **common, "valid_result": all(validity_checks.values()), "primary_claim_supported": all(claim_checks.values()), "selected_condition": "H0", "rerun_allowed": False}
    fields = ["seed", "clip_id", "path", "condition", "gold_state", "final_prediction", "final_state", "commits", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"]
    with (output / "per_path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(output / "predictions.npz", train_ids=train_ids, test_ids=test_ids, train_labels=train_labels, test_labels=test_labels, **{f"train_oof_{key}": value for key, value in train_predictions.items()}, **{f"test_{key}": value for key, value in test_predictions.items()}, H0_risk=np.asarray(risks))
    torch.save({"architecture": snapshot["models"]["risk_predictor"], "models": models}, output / "models.pt")
    matrix.write_json(output / "metrics.json", metrics); matrix.write_json(output / "decision.json", decision); matrix.write_json(output / "costs.json", {"schema": "amac-chsimsv2-test-costs-v1", "status": "completed", **common, "external_api_usd": 0.0, "wall_seconds": time.time() - started})
    artifacts = ["metrics.json", "decision.json", "costs.json", "per_path.csv", "predictions.npz", "models.pt"]
    matrix.write_json(output / "manifest.json", {"schema": "amac-chsimsv2-test-manifest-v1", "status": "completed_unvalidated", **common, "dataset_sha256": sha256_file(dataset_path), "used_splits": ["train", "test"], "selection_split_used": False, "counts": {"seeds": len(seeds), "test_clips": len(test_ids), "paths_per_clip": len(orders), "conditions": len(conditions), "rows": len(all_rows)}, "artifacts": {name: sha256_file(output / name) for name in artifacts}})
    print(json.dumps({"run_id": output.name, "valid_result": decision["valid_result"], "primary_claim_supported": decision["primary_claim_supported"], "claim_checks": claim_checks}, ensure_ascii=False))


if __name__ == "__main__": main()
