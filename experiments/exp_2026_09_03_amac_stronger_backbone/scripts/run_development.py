#!/usr/bin/env python3
import argparse, copy, csv, hashlib, importlib.util, json, pickle, time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

class MaskedFusion(torch.nn.Module):
    def __init__(self, dims, config):
        super().__init__(); width = config["token_dim"]
        self.projections = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(dim, width), torch.nn.GELU(), torch.nn.LayerNorm(width)) for dim in dims])
        self.modality_embedding = torch.nn.Parameter(torch.empty(3, width)); torch.nn.init.normal_(self.modality_embedding, std=0.02)
        layer = torch.nn.TransformerEncoderLayer(d_model=width, nhead=config["attention_heads"], dim_feedforward=config["feedforward_dim"], dropout=config["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=config["transformer_layers"])
        self.head = torch.nn.Sequential(torch.nn.Linear(width + 3, config["fusion_hidden"]), torch.nn.GELU(), torch.nn.Dropout(config["dropout"]), torch.nn.Linear(config["fusion_hidden"], 1))

    def forward(self, modalities, mask):
        tokens = torch.stack([project(value) for project, value in zip(self.projections, modalities)], dim=1) + self.modality_embedding.unsqueeze(0)
        tokens = tokens * mask.unsqueeze(-1)
        encoded = self.transformer(tokens, src_key_padding_mask=~mask.bool())
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(torch.cat((pooled, mask), dim=1)).squeeze(1)

def fit_scalers(pooled, indices):
    result = {}
    for modality, values in pooled.items():
        sample = values[indices].astype(np.float64); mean = sample.mean(axis=0); scale = sample.std(axis=0); scale[scale < 1e-8] = 1.0
        result[modality] = (mean.astype(np.float32), scale.astype(np.float32))
    return result

def transform(pooled, scalers, indices):
    result = []
    for modality in ("T", "A", "V"):
        mean, scale = scalers[modality]; values = ((pooled[modality][indices] - mean) / scale).astype(np.float32)
        result.append(torch.from_numpy(np.clip(np.nan_to_num(values), -20.0, 20.0)))
    return result

def subset_masks(subsets): return torch.tensor([[float(m in subset) for m in ("T", "A", "V")] for subset in subsets], dtype=torch.float32)

def train_model(pooled, labels, indices, subsets, model_config, training, seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    scalers = fit_scalers(pooled, indices); x = transform(pooled, scalers, indices); y = torch.from_numpy(labels[indices].astype(np.float32))
    model = MaskedFusion([value.shape[1] for value in x], model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["backbone_learning_rate"], weight_decay=training["backbone_weight_decay"])
    masks = subset_masks(subsets); rng = np.random.default_rng(seed); final_loss = None; model.train()
    for _ in range(training["backbone_epochs"]):
        order = rng.permutation(len(indices)); losses = []
        for start in range(0, len(order), training["backbone_batch_size"]):
            batch = order[start:start + training["backbone_batch_size"]]; size = len(batch)
            expanded = [value[batch].unsqueeze(1).expand(size, len(subsets), value.shape[1]).reshape(size * len(subsets), value.shape[1]) for value in x]
            batch_masks = masks.unsqueeze(0).expand(size, len(subsets), 3).reshape(size * len(subsets), 3)
            targets = y[batch].unsqueeze(1).expand(size, len(subsets)).reshape(-1)
            optimizer.zero_grad(); loss = torch.nn.functional.smooth_l1_loss(model(expanded, batch_masks), targets); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        final_loss = float(np.mean(losses))
    return model, scalers, final_loss

def predict(model, scalers, pooled, indices, subsets, batch_size=256):
    x = transform(pooled, scalers, indices); masks = subset_masks(subsets); result = {subset: [] for subset in subsets}; model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            values = [item[start:start + batch_size] for item in x]
            for subset_index, subset in enumerate(subsets):
                mask = masks[subset_index].unsqueeze(0).expand(len(values[0]), 3); result[subset].append(model(values, mask).numpy())
    return {key: np.concatenate(value).astype(np.float32) for key, value in result.items()}

def cross_fitted_predictions(pooled, labels, groups, subsets, config):
    predictions = {subset: np.empty(len(labels), dtype=np.float32) for subset in subsets}; models = []
    splitter = GroupKFold(n_splits=config["sampling"]["group_folds"])
    for fold, (fit_index, holdout_index) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        model, scalers, loss = train_model(pooled, labels, fit_index, subsets, config["models"]["treatment"], config["training"], config["sampling"]["backbone_seed"] + fold)
        observed = predict(model, scalers, pooled, holdout_index, subsets)
        for subset in subsets: predictions[subset][holdout_index] = observed[subset]
        models.append({"fold": fold, "fit_groups": sorted(set(groups[fit_index])), "holdout_groups": sorted(set(groups[holdout_index])), "loss": loss, "state_dict": copy.deepcopy(model.state_dict()), "scalers": scalers})
    return predictions, models

def event_score_map(metadata, values): return {(item[0], item[1], item[2]): float(value) for item, value in zip(metadata, values)}

def select_records(base, ids, labels, predictions, orders, scores, allowed_groups=None):
    records = base.create_records(ids, labels, predictions, orders, scores)
    return records if allowed_groups is None else [record for record in records if base.source_group(record[0]) in allowed_groups]

def evaluate_records(base, records, condition, parameters, delegated=None):
    rows = []
    for clip_id, order, label, predictions, scores in records:
        row = base.path_row(clip_id, order, delegated or condition, label, predictions, scores, parameters.get("threshold", 0.5), parameters.get("margin", 0.0)); row["policy"] = condition; rows.append(row)
    return rows

def aggregate(rows):
    commits = sum(row["prefinal_commits"] for row in rows); errors = sum(row["prefinal_errors"] for row in rows)
    return {"paths": len(rows), "prefinal_commits": commits, "prefinal_errors": errors, "prefinal_committed_error_rate": errors / commits if commits else None, "committed_revision_rate": sum(row["revisions"] for row in rows) / len(rows), "premature_exposure_rate": sum(row["premature"] for row in rows) / len(rows), "stage_two_coverage": sum(row["stage2_covered"] for row in rows) / len(rows), "time_to_first_commit": sum(row["time_to_first"] for row in rows) / len(rows), "final_state_identity": sum(row["final_identity"] for row in rows) / len(rows)}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    started = time.time(); snapshot_path = Path(args.parameter_snapshot).resolve(); output = Path(args.condition_dir).resolve()
    if output.exists(): raise RuntimeError("实验输出目录已存在，禁止覆盖或续跑")
    config = json.loads(snapshot_path.read_text(encoding="utf-8")); root = Path(__file__).resolve().parents[3]
    if config["status"] != "frozen": raise RuntimeError("参数快照未冻结")
    for key in ("runner", "validator", "base_runner", "matrix_runner"):
        if sha256(root / config["environment"][f"{key}_path"]) != config["environment"][f"{key}_sha256"]: raise RuntimeError(f"{key} 哈希不一致")
    dataset_path = root / config["dataset"]["path"]
    if sha256(dataset_path) != config["dataset"]["sha256"]: raise RuntimeError("数据集哈希不一致")
    reference_path = root / config["models"]["control"]["artifact"]
    if sha256(reference_path) != config["models"]["control"]["artifact_sha256"]: raise RuntimeError("Ridge 参考预测哈希不一致")
    output.mkdir(parents=True); torch.set_num_threads(config["resources"]["max_workers"])
    base = load_module("amac_base", root / config["environment"]["base_runner_path"]); matrix = load_module("amac_matrix", root / config["environment"]["matrix_runner_path"])
    with dataset_path.open("rb") as handle: dataset = pickle.load(handle)
    train = dataset[config["dataset"]["training_split"]]; valid = dataset[config["dataset"]["evaluation_split"]]
    train_ids = np.asarray(train["id"]).astype(str); valid_ids = np.asarray(valid["id"]).astype(str)
    train_labels = np.asarray(train["regression_labels"], dtype=np.float32).reshape(-1); valid_labels = np.asarray(valid["regression_labels"], dtype=np.float32).reshape(-1)
    if len(train_ids) != config["dataset"]["expected_train"] or len(valid_ids) != config["dataset"]["expected_evaluation"]: raise RuntimeError("数据划分样本数错误")
    if set(train_ids) & set(valid_ids): raise RuntimeError("训练与验证 clip 身份交叉")
    train_pooled = {"T": base.pool_modality(train["text"]), "A": base.pool_modality(train["audio"]), "V": base.pool_modality(train["vision"])}
    valid_pooled = {"T": base.pool_modality(valid["text"]), "A": base.pool_modality(valid["audio"]), "V": base.pool_modality(valid["vision"])}
    del dataset, train, valid
    groups = np.asarray([base.source_group(value) for value in train_ids]); subsets = config["sampling"]["subsets"]
    train_predictions, fold_models = cross_fitted_predictions(train_pooled, train_labels, groups, subsets, config)
    full_model, full_scalers, full_loss = train_model(train_pooled, train_labels, np.arange(len(train_ids)), subsets, config["models"]["treatment"], config["training"], config["sampling"]["backbone_seed"] + 100)
    valid_predictions = predict(full_model, full_scalers, valid_pooled, np.arange(len(valid_ids)), subsets)
    if any(not np.isfinite(value).all() for mapping in (train_predictions, valid_predictions) for value in mapping.values()): raise RuntimeError("基础预测含非有限值")
    with np.load(reference_path, allow_pickle=False) as reference:
        if not np.array_equal(reference["train_ids"].astype(str), train_ids) or not np.array_equal(reference["valid_ids"].astype(str), valid_ids): raise RuntimeError("Ridge 参考预测样本顺序不一致")
        reference_predictions = {subset: reference[f"valid_{subset}"].copy() for subset in subsets}
    terminal = {"masked_fusion": {subset: matrix.official_metrics(valid_predictions[subset], valid_labels) for subset in subsets}, "ridge_reference": {subset: matrix.official_metrics(reference_predictions[subset], valid_labels) for subset in subsets}}
    orders = config["sampling"]["arrival_orders"]
    train_features, targets, train_metadata = base.build_events(train_ids, train_labels, train_predictions, base.modality_quality(train_pooled), orders)
    valid_features, _, valid_metadata = base.build_events(valid_ids, valid_labels, valid_predictions, base.modality_quality(valid_pooled), orders)
    for values in (train_features, valid_features): values[:, config["preprocessing"]["quality_feature_start"]:] = 0.0; values[:, config["preprocessing"]["history_feature_indices"]] = 0.0
    targets[:, 1] = targets[:, 0]
    risk_config = dict(config["models"]["risk_predictor"]); risk_config.update({"epochs": config["training"]["risk_epochs"], "batch_size": config["training"]["risk_batch_size"], "learning_rate": config["training"]["risk_learning_rate"], "weight_decay": config["training"]["risk_weight_decay"], "patience": config["training"]["risk_patience"]})
    all_rows, per_seed, risk_models = [], {}, {}; calibration = {group for *_, group in train_metadata if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0}
    for seed in config["sampling"]["risk_seeds"]:
        np.random.seed(seed); torch.manual_seed(seed)
        model, scaler, _, _, training_info = base.train_amac(train_features, targets, train_metadata, risk_config, seed)
        train_scores, _ = base.predict_scores(model, scaler, train_features); valid_scores, _ = base.predict_scores(model, scaler, valid_features)
        train_records = select_records(base, train_ids, train_labels, train_predictions, orders, event_score_map(train_metadata, train_scores), calibration)
        valid_records = select_records(base, valid_ids, valid_labels, valid_predictions, orders, event_score_map(valid_metadata, valid_scores))
        _, b4_parameters = base.tune_policy(train_records, "B4", config["policy"]["threshold_grid"], config["policy"]["margin_grid"], config["policy"]["stage_two_coverage_target"])
        _, h0_parameters = base.tune_policy(train_records, "M0", config["policy"]["threshold_grid"], config["policy"]["margin_grid"], config["policy"]["stage_two_coverage_target"])
        seed_rows = evaluate_records(base, valid_records, "B0", {}) + evaluate_records(base, valid_records, "B1", {}) + evaluate_records(base, valid_records, "B4", b4_parameters) + evaluate_records(base, valid_records, "H0", h0_parameters, "M0")
        for row in seed_rows: row["seed"] = seed
        policy_metrics = {condition: aggregate([row for row in seed_rows if row["policy"] == condition]) for condition in config["policy"]["reported"]}
        bootstrap_rows = [dict(row, policy="M0" if row["policy"] == "H0" else row["policy"]) for row in seed_rows if row["policy"] in ("B4", "H0")]
        intervals = base.paired_bootstrap(bootstrap_rows, config["statistics"]["bootstrap_repetitions"], seed * 1009)
        effect = {"error_reduction": policy_metrics["B4"]["prefinal_committed_error_rate"] - policy_metrics["H0"]["prefinal_committed_error_rate"], "revision_relative_reduction": (policy_metrics["B4"]["committed_revision_rate"] - policy_metrics["H0"]["committed_revision_rate"]) / policy_metrics["B4"]["committed_revision_rate"], **intervals}
        per_seed[str(seed)] = {"policies": policy_metrics, "comparison": effect, "parameters": {"B4": b4_parameters, "H0": h0_parameters}, "training": training_info}
        risk_models[str(seed)] = {"state_dict": copy.deepcopy(model.state_dict()), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}; all_rows.extend(seed_rows)
    gates = config["gates"]
    checks = {"terminal_mae": terminal["masked_fusion"]["TAV"]["MAE"] - terminal["ridge_reference"]["TAV"]["MAE"] <= gates["terminal_mae_degradation_max"], "coverage": all(per_seed[str(seed)]["policies"]["H0"]["stage_two_coverage"] >= gates["coverage_min"] for seed in config["sampling"]["risk_seeds"]), "error_effect": all(per_seed[str(seed)]["comparison"]["error_reduction"] >= gates["vs_b4_error_reduction_min"] for seed in config["sampling"]["risk_seeds"]), "revision_effect": all(per_seed[str(seed)]["comparison"]["revision_relative_reduction"] >= gates["vs_b4_revision_relative_min"] for seed in config["sampling"]["risk_seeds"]), "intervals": all(per_seed[str(seed)]["comparison"]["error_absolute_reduction_ci95"][0] > gates["bootstrap_lower_bound_min"] and per_seed[str(seed)]["comparison"]["revision_relative_reduction_ci95"][0] > gates["bootstrap_lower_bound_min"] for seed in config["sampling"]["risk_seeds"])}
    validity = {"train_valid_disjoint": not bool(set(train_ids) & set(valid_ids)), "finite_predictions": all(np.isfinite(value).all() for mapping in (train_predictions, valid_predictions) for value in mapping.values()), "final_identity": all(per_seed[str(seed)]["policies"][condition]["final_state_identity"] == gates["final_identity"] for seed in config["sampling"]["risk_seeds"] for condition in config["policy"]["reported"])}
    common = {"run_id": output.name, "parameter_snapshot_sha256": sha256(snapshot_path), "protocol_version": config["environment"]["protocol_version"]}
    metrics_document = {"schema": "amac-masked-fusion-development-metrics-v1", "status": "completed", **common, "terminal": terminal, "per_seed": per_seed, "validity_checks": validity, "performance_checks": checks}
    decision = {"schema": "amac-masked-fusion-development-decision-v1", "status": "completed", **common, "valid_result": all(validity.values()), "continue_to_formal_test": all(validity.values()) and all(checks.values()), "paper_evidence": False}
    fields = ["seed", "clip_id", "path", "policy", "gold_state", "final_prediction", "final_state", "commits", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"]
    with (output / "per_path.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(output / "predictions.npz", train_ids=train_ids, valid_ids=valid_ids, train_labels=train_labels, valid_labels=valid_labels, **{f"train_oof_{key}": value for key, value in train_predictions.items()}, **{f"valid_{key}": value for key, value in valid_predictions.items()})
    torch.save({"backbone": {"folds": fold_models, "full_state_dict": full_model.state_dict(), "full_scalers": full_scalers, "full_loss": full_loss}, "risk_models": risk_models}, output / "models.pt")
    write_json(output / "metrics.json", metrics_document); write_json(output / "decision.json", decision); write_json(output / "costs.json", {"schema": "amac-masked-fusion-development-costs-v1", "status": "completed", **common, "external_api_usd": 0.0, "wall_seconds": time.time() - started})
    artifacts = ["metrics.json", "decision.json", "costs.json", "per_path.csv", "predictions.npz", "models.pt"]
    write_json(output / "manifest.json", {"schema": "amac-masked-fusion-development-manifest-v1", "status": "completed_unvalidated", **common, "dataset_sha256": sha256(dataset_path), "used_splits": ["train", "valid"], "forbidden_split_indexed": False, "counts": {"train": len(train_ids), "valid": len(valid_ids), "risk_seeds": len(config["sampling"]["risk_seeds"]), "paths_per_clip": len(orders), "conditions": len(config["policy"]["reported"]), "rows": len(all_rows)}, "artifacts": {name: sha256(output / name) for name in artifacts}})
    print(json.dumps({"run_id": output.name, "valid_result": decision["valid_result"], "continue_to_formal_test": decision["continue_to_formal_test"], "performance_checks": checks}, ensure_ascii=False))

if __name__ == "__main__": main()
