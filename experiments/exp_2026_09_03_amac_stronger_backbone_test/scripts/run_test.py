#!/usr/bin/env python3
import argparse, copy, csv, hashlib, importlib.util, json, pickle, time
from pathlib import Path
import numpy as np
import torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def summarize(rows, conditions, aggregate): return {condition: aggregate([row for row in rows if row["policy"] == condition]) for condition in conditions}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    started = time.time(); snapshot_path = Path(args.parameter_snapshot).resolve(); output = Path(args.condition_dir).resolve(); root = Path(__file__).resolve().parents[3]
    if output.exists(): raise RuntimeError("正式测试输出目录已存在，禁止覆盖或续跑")
    config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen": raise RuntimeError("参数快照未冻结")
    for key in ("runner", "validator", "development_runner", "base_runner", "matrix_runner", "governor_profile"):
        if sha256(root / config["environment"][f"{key}_path"]) != config["environment"][f"{key}_sha256"]: raise RuntimeError(f"{key} 哈希不一致")
    for key in ("decision", "validator"):
        path = root / config["development"][key]
        if sha256(path) != config["development"][f"{key}_sha256"]: raise RuntimeError(f"开发阶段 {key} 哈希不一致")
    if json.loads((root / config["development"]["validator"]).read_text())["status"] != "passed": raise RuntimeError("开发阶段未通过独立验证")
    dataset_path = root / config["dataset"]["path"]
    if sha256(dataset_path) != config["dataset"]["sha256"]: raise RuntimeError("数据集哈希不一致")
    reference_path = root / config["models"]["control"]["artifact"]
    if sha256(reference_path) != config["models"]["control"]["artifact_sha256"]: raise RuntimeError("Ridge 参考预测哈希不一致")
    output.mkdir(parents=True); torch.set_num_threads(config["resources"]["max_workers"])
    dev = load_module("masked_fusion_dev", root / config["environment"]["development_runner_path"]); base = load_module("amac_base", root / config["environment"]["base_runner_path"]); matrix = load_module("amac_matrix", root / config["environment"]["matrix_runner_path"])
    with dataset_path.open("rb") as handle: dataset = pickle.load(handle)
    train = dataset[config["dataset"]["training_split"]]; test = dataset[config["dataset"]["evaluation_split"]]
    train_ids = np.asarray(train["id"]).astype(str); test_ids = np.asarray(test["id"]).astype(str); train_labels = np.asarray(train["regression_labels"], dtype=np.float32).reshape(-1); test_labels = np.asarray(test["regression_labels"], dtype=np.float32).reshape(-1)
    if len(train_ids) != config["dataset"]["expected_train"] or len(test_ids) != config["dataset"]["expected_evaluation"]: raise RuntimeError("数据划分样本数错误")
    if set(train_ids) & set(test_ids): raise RuntimeError("训练与测试 clip 身份交叉")
    train_pooled = {"T": base.pool_modality(train["text"]), "A": base.pool_modality(train["audio"]), "V": base.pool_modality(train["vision"])}; test_pooled = {"T": base.pool_modality(test["text"]), "A": base.pool_modality(test["audio"]), "V": base.pool_modality(test["vision"])}
    del dataset, train, test
    groups = np.asarray([base.source_group(value) for value in train_ids]); subsets = config["sampling"]["subsets"]
    train_predictions, fold_models = dev.cross_fitted_predictions(train_pooled, train_labels, groups, subsets, config)
    full_model, full_scalers, full_loss = dev.train_model(train_pooled, train_labels, np.arange(len(train_ids)), subsets, config["models"]["treatment"], config["training"], config["sampling"]["backbone_seed"] + 100)
    test_predictions = dev.predict(full_model, full_scalers, test_pooled, np.arange(len(test_ids)), subsets)
    if any(not np.isfinite(value).all() for mapping in (train_predictions, test_predictions) for value in mapping.values()): raise RuntimeError("基础预测含非有限值")
    with np.load(reference_path, allow_pickle=False) as reference:
        if not np.array_equal(reference["train_ids"].astype(str), train_ids) or not np.array_equal(reference["test_ids"].astype(str), test_ids): raise RuntimeError("Ridge 参考预测样本顺序不一致")
        reference_predictions = {subset: reference[f"test_{subset}"].copy() for subset in subsets}
    terminal = {"masked_fusion": {subset: matrix.official_metrics(test_predictions[subset], test_labels) for subset in subsets}, "ridge_reference": {subset: matrix.official_metrics(reference_predictions[subset], test_labels) for subset in subsets}}
    orders = config["sampling"]["arrival_orders"]; conditions = config["policy"]["reported"]
    train_features, targets, train_metadata = base.build_events(train_ids, train_labels, train_predictions, base.modality_quality(train_pooled), orders); test_features, _, test_metadata = base.build_events(test_ids, test_labels, test_predictions, base.modality_quality(test_pooled), orders)
    for values in (train_features, test_features): values[:, config["preprocessing"]["quality_feature_start"]:] = 0.0; values[:, config["preprocessing"]["history_feature_indices"]] = 0.0
    targets[:, 1] = targets[:, 0]
    risk_config = dict(config["models"]["risk_predictor"]); risk_config.update({"epochs": config["training"]["risk_epochs"], "batch_size": config["training"]["risk_batch_size"], "learning_rate": config["training"]["risk_learning_rate"], "weight_decay": config["training"]["risk_weight_decay"], "patience": config["training"]["risk_patience"]})
    calibration = {group for *_, group in train_metadata if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0}; source_disjoint_ids = {clip_id for clip_id in test_ids if base.source_group(clip_id) not in set(groups)}
    all_rows, per_seed, risk_models = [], {}, {}
    for seed in config["sampling"]["risk_seeds"]:
        np.random.seed(seed); torch.manual_seed(seed)
        model, scaler, _, _, training_info = base.train_amac(train_features, targets, train_metadata, risk_config, seed)
        train_scores, _ = base.predict_scores(model, scaler, train_features); test_scores, _ = base.predict_scores(model, scaler, test_features)
        train_records = dev.select_records(base, train_ids, train_labels, train_predictions, orders, dev.event_score_map(train_metadata, train_scores), calibration); test_records = dev.select_records(base, test_ids, test_labels, test_predictions, orders, dev.event_score_map(test_metadata, test_scores))
        _, b2_parameters = base.tune_policy(train_records, "B2", config["policy"]["threshold_grid"], config["policy"]["margin_grid"], config["policy"]["stage_two_coverage_target"]); _, b4_parameters = base.tune_policy(train_records, "B4", config["policy"]["threshold_grid"], config["policy"]["margin_grid"], config["policy"]["stage_two_coverage_target"]); _, h0_parameters = base.tune_policy(train_records, "M0", config["policy"]["threshold_grid"], config["policy"]["margin_grid"], config["policy"]["stage_two_coverage_target"])
        seed_rows = []
        for condition in ("B0", "B1", "B3", "O1"): seed_rows += dev.evaluate_records(base, test_records, condition, {})
        seed_rows += dev.evaluate_records(base, test_records, "B2", b2_parameters) + dev.evaluate_records(base, test_records, "B4", b4_parameters) + dev.evaluate_records(base, test_records, "H0", h0_parameters, "M0")
        for row in seed_rows: row["seed"] = seed
        policy_metrics = summarize(seed_rows, conditions, dev.aggregate); bootstrap_rows = [dict(row, policy="M0" if row["policy"] == "H0" else row["policy"]) for row in seed_rows if row["policy"] in ("B4", "H0")]
        intervals = base.paired_bootstrap(bootstrap_rows, config["statistics"]["bootstrap_repetitions"], seed * 1009); effect = {"error_reduction": policy_metrics["B4"]["prefinal_committed_error_rate"] - policy_metrics["H0"]["prefinal_committed_error_rate"], "coverage_gap": abs(policy_metrics["B4"]["stage_two_coverage"] - policy_metrics["H0"]["stage_two_coverage"]), "revision_relative_reduction": (policy_metrics["B4"]["committed_revision_rate"] - policy_metrics["H0"]["committed_revision_rate"]) / policy_metrics["B4"]["committed_revision_rate"], **intervals}
        source_rows = [row for row in seed_rows if row["clip_id"] in source_disjoint_ids]; source_metrics = summarize(source_rows, conditions, dev.aggregate); source_bootstrap = [dict(row, policy="M0" if row["policy"] == "H0" else row["policy"]) for row in source_rows if row["policy"] in ("B4", "H0")]; source_intervals = base.paired_bootstrap(source_bootstrap, config["statistics"]["bootstrap_repetitions"], seed * 2017)
        source_effect = {"error_reduction": source_metrics["B4"]["prefinal_committed_error_rate"] - source_metrics["H0"]["prefinal_committed_error_rate"], "coverage_gap": abs(source_metrics["B4"]["stage_two_coverage"] - source_metrics["H0"]["stage_two_coverage"]), "revision_relative_reduction": (source_metrics["B4"]["committed_revision_rate"] - source_metrics["H0"]["committed_revision_rate"]) / source_metrics["B4"]["committed_revision_rate"], **source_intervals}
        per_seed[str(seed)] = {"policies": policy_metrics, "comparison": effect, "source_disjoint": {"clip_count": len(source_disjoint_ids), "policies": source_metrics, "comparison": source_effect}, "parameters": {"B2": b2_parameters, "B4": b4_parameters, "H0": h0_parameters}, "training": training_info}
        risk_models[str(seed)] = {"state_dict": copy.deepcopy(model.state_dict()), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}; all_rows.extend(seed_rows)
    gates = config["gates"]
    performance = {"terminal_mae": terminal["masked_fusion"]["TAV"]["MAE"] - terminal["ridge_reference"]["TAV"]["MAE"] <= gates["terminal_mae_degradation_max"], "coverage": all(per_seed[str(seed)]["policies"]["H0"]["stage_two_coverage"] >= gates["h0_coverage_min"] for seed in config["sampling"]["risk_seeds"]), "matched_coverage": all(per_seed[str(seed)]["comparison"]["coverage_gap"] <= gates["coverage_gap_max"] for seed in config["sampling"]["risk_seeds"]), "error_effect": all(per_seed[str(seed)]["comparison"]["error_reduction"] >= gates["vs_b4_error_reduction_min"] for seed in config["sampling"]["risk_seeds"]), "error_intervals": all(per_seed[str(seed)]["comparison"]["error_absolute_reduction_ci95"][0] > 0 for seed in config["sampling"]["risk_seeds"])}
    validity = {"train_test_disjoint": not bool(set(train_ids) & set(test_ids)), "finite_predictions": all(np.isfinite(value).all() for mapping in (train_predictions, test_predictions) for value in mapping.values()), "five_seeds": len(per_seed) == 5, "final_identity": all(per_seed[str(seed)]["policies"][condition]["final_state_identity"] == gates["final_identity"] for seed in config["sampling"]["risk_seeds"] for condition in conditions), "source_disjoint_nonempty": len(source_disjoint_ids) > 0}
    common = {"run_id": output.name, "parameter_snapshot_sha256": sha256(snapshot_path), "protocol_version": config["environment"]["protocol_version"]}; metrics_document = {"schema": "amac-masked-fusion-test-metrics-v1", "status": "completed", **common, "terminal": terminal, "source_disjoint_clip_count": len(source_disjoint_ids), "per_seed": per_seed, "validity_checks": validity, "performance_checks": performance}; decision = {"schema": "amac-masked-fusion-test-decision-v1", "status": "completed", **common, "valid_result": all(validity.values()), "primary_claim_supported": all(validity.values()) and all(performance.values()), "revision_claim": "exploratory_only", "rerun_allowed": False}
    fields = ["seed", "clip_id", "path", "policy", "gold_state", "final_prediction", "final_state", "commits", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"]
    with (output / "per_path.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(output / "predictions.npz", train_ids=train_ids, test_ids=test_ids, train_labels=train_labels, test_labels=test_labels, source_disjoint_ids=np.asarray(sorted(source_disjoint_ids)), **{f"train_oof_{key}": value for key, value in train_predictions.items()}, **{f"test_{key}": value for key, value in test_predictions.items()}); torch.save({"backbone": {"folds": fold_models, "full_state_dict": full_model.state_dict(), "full_scalers": full_scalers, "full_loss": full_loss}, "risk_models": risk_models}, output / "models.pt")
    write_json(output / "metrics.json", metrics_document); write_json(output / "decision.json", decision); write_json(output / "costs.json", {"schema": "amac-masked-fusion-test-costs-v1", "status": "completed", **common, "external_api_usd": 0.0, "wall_seconds": time.time() - started})
    artifacts = ["metrics.json", "decision.json", "costs.json", "per_path.csv", "predictions.npz", "models.pt"]; write_json(output / "manifest.json", {"schema": "amac-masked-fusion-test-manifest-v1", "status": "completed_unvalidated", **common, "dataset_sha256": sha256(dataset_path), "used_splits": ["train", "test"], "selection_split_indexed": False, "counts": {"train": len(train_ids), "test": len(test_ids), "source_disjoint": len(source_disjoint_ids), "risk_seeds": len(config["sampling"]["risk_seeds"]), "paths_per_clip": len(orders), "conditions": len(conditions), "rows": len(all_rows)}, "artifacts": {name: sha256(output / name) for name in artifacts}})
    print(json.dumps({"run_id": output.name, "valid_result": decision["valid_result"], "primary_claim_supported": decision["primary_claim_supported"], "performance_checks": performance}, ensure_ascii=False))

if __name__ == "__main__": main()
