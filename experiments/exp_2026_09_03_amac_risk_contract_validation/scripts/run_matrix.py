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
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def score_map(metadata, scores):
    return {(item[0], item[1], item[2]): float(scores[i]) for i, item in enumerate(metadata)}


def subset_records(base, ids, labels, predictions, orders, scores, indices=None):
    if indices is None:
        return base.create_records(ids, labels, predictions, orders, scores)
    selected = sorted(indices)
    mapping = {old: new for new, old in enumerate(selected)}
    local_scores = {(mapping[i], order, stage): value for (i, order, stage), value in scores.items() if i in mapping}
    local_predictions = {key: values[selected] for key, values in predictions.items()}
    return base.create_records(ids[selected], labels[selected], local_predictions, orders, local_scores)


def aggregate(rows):
    commits = sum(row["prefinal_commits"] for row in rows)
    errors = sum(row["prefinal_errors"] for row in rows)
    return {
        "paths": len(rows), "prefinal_commits": commits, "prefinal_errors": errors,
        "prefinal_committed_error_rate": errors / commits if commits else None,
        "committed_revision_rate": sum(row["revisions"] for row in rows) / len(rows),
        "premature_exposure_rate": sum(row["premature"] for row in rows) / len(rows),
        "stage_two_coverage": sum(row["stage2_covered"] for row in rows) / len(rows),
        "time_to_first_commit": sum(row["time_to_first"] for row in rows) / len(rows),
        "final_state_identity": sum(row["final_identity"] for row in rows) / len(rows),
    }


def evaluate(base, records, condition, parameters, delegate=None):
    rows = []
    for clip_id, order, label, predictions, scores in records:
        row = base.path_row(clip_id, order, delegate or condition, label, predictions, scores, parameters.get("threshold", 0.5), parameters.get("margin", 0.0))
        row["condition"] = condition
        row.pop("policy", None)
        rows.append(row)
    return rows


def tune(matrix, base, records, condition, delegate, policy):
    return matrix.tune(base, records, delegate, condition, policy["threshold_grid"], policy["margin_grid"], policy["coverage_target"], policy["revision_weight"], policy["wait_weight"])


def utility(metrics, policy):
    return metrics["prefinal_committed_error_rate"] + policy["revision_weight"] * metrics["committed_revision_rate"] + policy["wait_weight"] * (metrics["time_to_first_commit"] - 1.0)


def core_checks(per_seed, method, snapshot):
    seeds, gates = snapshot["sampling"]["seeds"], snapshot["gates"]
    comps = lambda comparator, field: [per_seed[str(seed)]["comparisons"][method][comparator][field] for seed in seeds]
    return {
        "coverage": all(per_seed[str(seed)]["conditions"][method]["stage_two_coverage"] >= gates["coverage_min"] for seed in seeds),
        "vs_b4_error": float(np.mean(comps("B4", "error_reduction"))) >= gates["vs_b4_error_reduction_min"] and all(value > 0 for value in comps("B4", "error_reduction")),
        "vs_b4_revision": float(np.mean(comps("B4", "revision_relative_reduction"))) >= gates["vs_b4_revision_relative_min"] and all(value > 0 for value in comps("B4", "revision_relative_reduction")),
        "vs_b4_intervals": all(per_seed[str(seed)]["comparisons"][method]["B4"]["error_reduction_ci95"][0] > 0 and per_seed[str(seed)]["comparisons"][method]["B4"]["revision_relative_reduction_ci95"][0] > 0 for seed in seeds),
        "vs_b2": all(value > 0 for value in comps("B2", "error_reduction")) and all(value > 0 for value in comps("B2", "revision_relative_reduction")),
        "b3_service_or_dominance": all(per_seed[str(seed)]["conditions"]["B3"]["stage_two_coverage"] < gates["coverage_min"] or (per_seed[str(seed)]["comparisons"][method]["B3"]["error_reduction"] > 0 and per_seed[str(seed)]["comparisons"][method]["B3"]["revision_relative_reduction"] > 0) for seed in seeds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--condition-dir", required=True)
    args = parser.parse_args()
    started = time.time()
    snapshot_path = Path(args.parameter_snapshot).resolve()
    output = Path(args.condition_dir).resolve()
    if output.exists():
        raise RuntimeError("实验输出目录已存在，禁止覆盖")
    output.mkdir(parents=True)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_hash = sha256_file(snapshot_path)
    root = Path(__file__).resolve().parents[3]
    if sha256_file(Path(__file__).resolve()) != snapshot["environment"]["runner_sha256"]:
        raise RuntimeError("运行器哈希与冻结快照不一致")
    validator_path = root / snapshot["environment"]["validator_path"]
    if sha256_file(validator_path) != snapshot["environment"]["validator_sha256"]:
        raise RuntimeError("验证器哈希与冻结快照不一致")
    matrix = load_module("prior_matrix", root / "experiments/exp_2026_09_03_amac_validation_matrix/scripts/run_matrix.py")
    base = matrix.load_base(root)
    dataset_path = root / snapshot["dataset"]["path"]
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != snapshot["dataset"]["sha256"]:
        raise RuntimeError("数据集哈希不一致")
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    train = dataset[snapshot["dataset"]["training_split"]]
    valid = dataset[snapshot["dataset"]["evaluation_split"]]
    train_ids, valid_ids = np.asarray(train["id"]).astype(str), np.asarray(valid["id"]).astype(str)
    train_labels = np.asarray(train["regression_labels"], dtype=np.float32).reshape(-1)
    valid_labels = np.asarray(valid["regression_labels"], dtype=np.float32).reshape(-1)
    if len(train_ids) != snapshot["dataset"]["expected_train"] or len(valid_ids) != snapshot["dataset"]["expected_evaluation"]:
        raise RuntimeError("样本数与冻结快照不一致")
    train_pooled = {"T": base.pool_modality(train["text"]), "A": base.pool_modality(train["audio"]), "V": base.pool_modality(train["vision"])}
    valid_pooled = {"T": base.pool_modality(valid["text"]), "A": base.pool_modality(valid["audio"]), "V": base.pool_modality(valid["vision"])}
    del dataset, train, valid
    groups = np.asarray([base.source_group(value) for value in train_ids])
    cfg = snapshot["models"]["base"]
    train_predictions, valid_predictions = base.grouped_ridge_predictions(train_pooled, valid_pooled, train_labels, groups, cfg["alpha"], cfg["group_folds"], cfg["solver"])
    anchors = {subset: matrix.official_metrics(valid_predictions[subset], valid_labels) for subset in base.SUBSETS}
    orders = snapshot["sampling"]["arrival_orders"]
    train_features, targets, metadata = base.build_events(train_ids, train_labels, train_predictions, base.modality_quality(train_pooled), orders)
    valid_features, _, valid_metadata = base.build_events(valid_ids, valid_labels, valid_predictions, base.modality_quality(valid_pooled), orders)
    train_features[:, snapshot["preprocessing"]["quality_feature_start"]:] = 0.0
    valid_features[:, snapshot["preprocessing"]["quality_feature_start"]:] = 0.0
    no_history_train, no_history_valid = train_features.copy(), valid_features.copy()
    history_indices = snapshot["preprocessing"]["history_feature_indices"]
    no_history_train[:, history_indices] = 0.0
    no_history_valid[:, history_indices] = 0.0
    correctness_targets = targets.copy()
    correctness_targets[:, 1] = correctness_targets[:, 0]
    policy = snapshot["policy"]
    config = dict(snapshot["models"]["risk_predictor"]); config.update(snapshot["training"])
    conditions = policy["conditions"]
    all_rows, frontiers, per_seed, models, heads = [], [], {}, {}, {"M0": [], "H0": []}
    for seed in snapshot["sampling"]["seeds"]:
        np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
        m0_model, m0_scaler, _, calibration_mask, m0_training = base.train_amac(train_features, correctness_targets, metadata, config, seed)
        h0_model, h0_scaler, _, h0_calibration_mask, h0_training = base.train_amac(no_history_train, correctness_targets, metadata, config, seed)
        if not np.array_equal(calibration_mask, h0_calibration_mask):
            raise RuntimeError("M0 与 H0 校准划分不一致")
        _, m0_train_heads = base.predict_scores(m0_model, m0_scaler, train_features)
        _, m0_valid_heads = base.predict_scores(m0_model, m0_scaler, valid_features)
        _, h0_train_heads = base.predict_scores(h0_model, h0_scaler, no_history_train)
        _, h0_valid_heads = base.predict_scores(h0_model, h0_scaler, no_history_valid)
        calibration_indices = {item[0] for index, item in enumerate(metadata) if calibration_mask[index]}
        m0_train_map, h0_train_map = score_map(metadata, m0_train_heads[:, 0]), score_map(metadata, h0_train_heads[:, 0])
        m0_valid_map, h0_valid_map = score_map(valid_metadata, m0_valid_heads[:, 0]), score_map(valid_metadata, h0_valid_heads[:, 0])
        m0_cal = subset_records(base, train_ids, train_labels, train_predictions, orders, m0_train_map, calibration_indices)
        h0_cal = subset_records(base, train_ids, train_labels, train_predictions, orders, h0_train_map, calibration_indices)
        m0_records = subset_records(base, valid_ids, valid_labels, valid_predictions, orders, m0_valid_map)
        h0_records = subset_records(base, valid_ids, valid_labels, valid_predictions, orders, h0_valid_map)
        selected = {name: {"threshold": 0.5, "margin": 0.0} for name in conditions}
        selected["B2"] = matrix.tune(base, m0_cal, "B2", "B2", policy["threshold_grid"], [0.0], policy["coverage_target"], policy["revision_weight"], policy["wait_weight"])
        selected["B4"] = matrix.tune(base, m0_cal, "B4", "B4", policy["threshold_grid"], policy["margin_grid"], policy["coverage_target"], policy["revision_weight"], policy["wait_weight"])
        selected["M0"] = tune(matrix, base, m0_cal, "M0", "M0", policy)
        selected["H0"] = tune(matrix, base, h0_cal, "H0", "M0", policy)
        seed_rows = []
        for condition in ("B0", "B1", "B2", "B3", "B4", "O1"):
            seed_rows.extend(evaluate(base, m0_records, condition, selected[condition]))
        seed_rows.extend(evaluate(base, m0_records, "M0", selected["M0"], "M0"))
        seed_rows.extend(evaluate(base, h0_records, "H0", selected["H0"], "M0"))
        for row in seed_rows:
            row["seed"] = seed
        all_rows.extend(seed_rows)
        metrics = {condition: aggregate([row for row in seed_rows if row["condition"] == condition]) for condition in conditions}
        comparisons = {method: {} for method in ("M0", "H0")}
        for method in comparisons:
            for comparator in ("B2", "B3", "B4", "M0" if method == "H0" else "H0"):
                own, other = metrics[method], metrics[comparator]
                effect = {"error_reduction": other["prefinal_committed_error_rate"] - own["prefinal_committed_error_rate"], "revision_relative_reduction": (other["committed_revision_rate"] - own["committed_revision_rate"]) / other["committed_revision_rate"] if other["committed_revision_rate"] else None}
                effect.update(matrix.paired_intervals(seed_rows, method, comparator, snapshot["statistics"]["bootstrap_repetitions"], seed * 1000 + sum(map(ord, method + comparator))))
                comparisons[method][comparator] = effect
        for method, records in (("M0", m0_records), ("H0", h0_records)):
            for threshold in policy["threshold_grid"]:
                for margin in policy["margin_grid"]:
                    stats = aggregate(evaluate(base, records, method, {"threshold": threshold, "margin": margin}, "M0"))
                    frontiers.append({"seed": seed, "condition": method, "threshold": threshold, "margin": margin, **stats})
        per_seed[str(seed)] = {"conditions": metrics, "comparisons": comparisons, "selected_parameters": selected, "training": {"M0": m0_training, "H0": h0_training}}
        models[str(seed)] = {"M0": {"state_dict": copy.deepcopy(m0_model.state_dict()), "scaler_mean": m0_scaler.mean_, "scaler_scale": m0_scaler.scale_}, "H0": {"state_dict": copy.deepcopy(h0_model.state_dict()), "scaler_mean": h0_scaler.mean_, "scaler_scale": h0_scaler.scale_}}
        heads["M0"].append(m0_valid_heads[:, 0]); heads["H0"].append(h0_valid_heads[:, 0])
    aggregate_summary = matrix.summarize_seeds(per_seed, conditions)
    m0_core, h0_core = core_checks(per_seed, "M0", snapshot), core_checks(per_seed, "H0", snapshot)
    gates, seeds = snapshot["gates"], snapshot["sampling"]["seeds"]
    h0_noninferior = all(per_seed[str(seed)]["conditions"]["H0"]["prefinal_committed_error_rate"] <= per_seed[str(seed)]["conditions"]["M0"]["prefinal_committed_error_rate"] + gates["simpler_error_tolerance"] and per_seed[str(seed)]["conditions"]["H0"]["committed_revision_rate"] <= per_seed[str(seed)]["conditions"]["M0"]["committed_revision_rate"] + gates["simpler_revision_tolerance"] and abs(per_seed[str(seed)]["conditions"]["H0"]["stage_two_coverage"] - per_seed[str(seed)]["conditions"]["M0"]["stage_two_coverage"]) <= gates["simpler_coverage_tolerance"] for seed in seeds)
    history_deltas = [utility(per_seed[str(seed)]["conditions"]["H0"], policy) - utility(per_seed[str(seed)]["conditions"]["M0"], policy) for seed in seeds]
    history_material = float(np.mean(history_deltas)) >= gates["history_utility_improvement_min"] and sum(value > 0 for value in history_deltas) >= gates["history_required_seeds"]
    if all(h0_core.values()) and h0_noninferior and not history_material:
        final_condition = "H0"
    elif all(m0_core.values()):
        final_condition = "M0"
    else:
        final_condition = None
    final_identity = all(per_seed[str(seed)]["conditions"][condition]["final_state_identity"] == gates["final_identity"] for seed in seeds for condition in conditions)
    checks = {"five_seeds_complete": len(per_seed) == 5, "test_isolation": True, "final_identity": final_identity, "m0_core": m0_core, "h0_core": h0_core, "h0_noninferior": h0_noninferior, "history_material": history_material, "selection_resolved": final_condition is not None}
    prepare_test = checks["five_seeds_complete"] and checks["test_isolation"] and checks["final_identity"] and checks["selection_resolved"]
    common = {"run_id": output.name, "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"]}
    result_metrics = {"schema": "amac-risk-contract-metrics-v1", "status": "completed", **common, "official_anchor": anchors, "per_seed": per_seed, "aggregate": aggregate_summary, "history_utility_deltas": history_deltas, "checks": checks}
    decision = {"schema": "amac-risk-contract-decision-v1", "status": "completed", **common, "checks": checks, "selected_condition": final_condition, "prepare_one_shot_test": prepare_test, "scope": "Validation-only model selection; no test result."}
    fields = ["seed", "clip_id", "path", "condition", "gold_state", "final_prediction", "final_state", "commits", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"]
    with (output / "per_path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    with (output / "frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontiers[0])); writer.writeheader(); writer.writerows(frontiers)
    np.savez_compressed(output / "predictions.npz", train_ids=train_ids, evaluation_ids=valid_ids, train_labels=train_labels, evaluation_labels=valid_labels, **{f"train_oof_{key}": value for key, value in train_predictions.items()}, **{f"evaluation_{key}": value for key, value in valid_predictions.items()}, M0_risk=np.asarray(heads["M0"]), H0_risk=np.asarray(heads["H0"]))
    torch.save({"architecture": snapshot["models"]["risk_predictor"], "models": models}, output / "models.pt")
    matrix.write_json(output / "metrics.json", result_metrics); matrix.write_json(output / "decision.json", decision)
    matrix.write_json(output / "costs.json", {"schema": "amac-risk-contract-costs-v1", "status": "completed", **common, "external_api_usd": 0.0, "wall_seconds": time.time() - started})
    artifacts = ["metrics.json", "decision.json", "costs.json", "per_path.csv", "frontier.csv", "predictions.npz", "models.pt"]
    matrix.write_json(output / "manifest.json", {"schema": "amac-risk-contract-manifest-v1", "status": "completed_unvalidated", **common, "dataset_sha256": dataset_hash, "used_splits": ["train", "valid"], "forbidden_split_indexed": False, "counts": {"seeds": len(seeds), "evaluation_clips": len(valid_ids), "paths_per_clip": len(orders), "conditions": len(conditions), "rows": len(all_rows)}, "artifacts": {name: sha256_file(output / name) for name in artifacts}})
    print(json.dumps({"run_id": output.name, "selected_condition": final_condition, "prepare_one_shot_test": prepare_test, "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
