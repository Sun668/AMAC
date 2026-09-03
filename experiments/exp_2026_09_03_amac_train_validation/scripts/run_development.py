#!/usr/bin/env python3
import argparse
import copy
import csv
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SUBSETS = ("T", "A", "V", "TA", "TV", "AV", "TAV")
MODALITIES = ("T", "A", "V")
POLICIES = ("B0", "B1", "B2", "B3", "B4", "M0", "O1")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_subset(chars):
    present = set(chars)
    return "".join(modality for modality in MODALITIES if modality in present)


def source_group(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).split("$_$", 1)[0]


def affect_state(values, boundaries=(-0.1, 0.1)):
    values = np.asarray(values)
    return np.where(values < boundaries[0], -1, np.where(values > boundaries[1], 1, 0)).astype(np.int8)


def fixed_confidence(value):
    value = float(value)
    if value < -0.1:
        return min(1.0, (-0.1 - value) / 0.9)
    if value > 0.1:
        return min(1.0, (value - 0.1) / 0.9)
    return min(1.0, (0.1 - abs(value)) / 0.1)


def pool_modality(array):
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(array, -10000.0, 10000.0, out=array)
    return np.concatenate((array.mean(axis=1), array.std(axis=1)), axis=1).astype(np.float32)


def stable_transform(scaler, values):
    transformed = scaler.transform(values).astype(np.float32)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=20.0, neginf=-20.0)
    return np.clip(transformed, -20.0, 20.0)


def stable_linear_predict(model, values):
    values = np.asarray(values, dtype=np.float64)
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    return np.einsum("ij,j->i", values, coefficients, optimize=False) + float(model.intercept_)


def build_subset_features(pooled, subset):
    return np.concatenate([pooled[m] for m in MODALITIES if m in subset], axis=1)


def grouped_ridge_predictions(train_pooled, valid_pooled, labels, groups, alpha, folds, solver):
    oof = {}
    valid = {}
    for subset in SUBSETS:
        x_train = build_subset_features(train_pooled, subset)
        x_valid = build_subset_features(valid_pooled, subset)
        subset_oof = np.empty(len(labels), dtype=np.float32)
        splitter = GroupKFold(n_splits=folds)
        for train_index, holdout_index in splitter.split(x_train, labels, groups):
            scaler = StandardScaler().fit(x_train[train_index])
            label_mean = float(np.mean(labels[train_index], dtype=np.float64))
            model = Ridge(alpha=alpha, solver=solver, fit_intercept=False).fit(stable_transform(scaler, x_train[train_index]).astype(np.float64), labels[train_index].astype(np.float64) - label_mean)
            subset_oof[holdout_index] = (stable_linear_predict(model, stable_transform(scaler, x_train[holdout_index])) + label_mean).astype(np.float32)
        scaler = StandardScaler().fit(x_train)
        label_mean = float(np.mean(labels, dtype=np.float64))
        model = Ridge(alpha=alpha, solver=solver, fit_intercept=False).fit(stable_transform(scaler, x_train).astype(np.float64), labels.astype(np.float64) - label_mean)
        oof[subset] = subset_oof
        valid[subset] = (stable_linear_predict(model, stable_transform(scaler, x_valid)) + label_mean).astype(np.float32)
        if not np.isfinite(oof[subset]).all() or not np.isfinite(valid[subset]).all():
            raise RuntimeError(f"{subset} 基础预测包含非有限值")
    return oof, valid


def modality_quality(pooled):
    result = {}
    for modality, values in pooled.items():
        half = values.shape[1] // 2
        mean_part = values[:, :half]
        std_part = values[:, half:]
        result[modality] = np.column_stack((
            np.linalg.norm(mean_part, axis=1) / max(1.0, np.sqrt(half)),
            np.mean(np.abs(std_part), axis=1),
        )).astype(np.float32)
    return result


def one_hot_state(state):
    return [float(state == -1), float(state == 0), float(state == 1)]


def event_features(index, order, stage, predictions, qualities):
    subsets = [canonical_subset(order[:position]) for position in (1, 2, 3)]
    history = [float(predictions[subset][index]) for subset in subsets[:stage + 1]]
    current = history[-1]
    previous = history[-2] if len(history) > 1 else 0.0
    current_state = int(affect_state([current])[0])
    previous_state = int(affect_state([previous])[0]) if len(history) > 1 else 99
    visible = set(subsets[stage])
    vector = [current, abs(current), previous, current - previous if len(history) > 1 else 0.0]
    vector.extend([float(np.mean(history)), float(np.std(history)), min(history), max(history)])
    vector.append(stage / 2.0)
    vector.extend([float(modality in visible) for modality in MODALITIES])
    vector.extend(one_hot_state(current_state))
    vector.extend(one_hot_state(previous_state) + [float(previous_state == 99)])
    for modality in MODALITIES:
        if modality in visible:
            vector.extend(qualities[modality][index].tolist())
        else:
            vector.extend([0.0, 0.0])
    return np.asarray(vector, dtype=np.float32)


def build_events(ids, labels, predictions, qualities, orders):
    features, correct_targets, stable_targets, metadata = [], [], [], []
    gold_states = affect_state(labels)
    for index, clip_id in enumerate(ids):
        for order_index, order in enumerate(orders):
            subsets = [canonical_subset(order[:position]) for position in (1, 2, 3)]
            path_states = [int(affect_state([predictions[subset][index]])[0]) for subset in subsets]
            for stage in (0, 1):
                features.append(event_features(index, order, stage, predictions, qualities))
                correct_targets.append(float(path_states[stage] == gold_states[index]))
                stable_targets.append(float(all(state == path_states[stage] for state in path_states[stage + 1:])))
                metadata.append((index, order_index, stage, str(clip_id), source_group(clip_id)))
    return (
        np.asarray(features, dtype=np.float32),
        np.column_stack((correct_targets, stable_targets)).astype(np.float32),
        metadata,
    )


class DualHeadMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[1], 2),
        )

    def forward(self, values):
        return self.network(values)


def train_amac(features, targets, metadata, config, seed):
    calibration_groups = {group for *_, group in metadata if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0}
    train_mask = np.asarray([item[-1] not in calibration_groups for item in metadata])
    calibration_mask = ~train_mask
    if not train_mask.any() or not calibration_mask.any():
        raise RuntimeError("AMAC 内部训练/校准分组为空")
    scaler = StandardScaler().fit(features[train_mask])
    x_train = torch.from_numpy(stable_transform(scaler, features[train_mask]))
    y_train = torch.from_numpy(targets[train_mask])
    x_calibration = torch.from_numpy(stable_transform(scaler, features[calibration_mask]))
    y_calibration = torch.from_numpy(targets[calibration_mask])
    model = DualHeadMLP(features.shape[1], config["hidden"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_function = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, -1
    for epoch in range(config["epochs"]):
        model.train()
        permutation = rng.permutation(len(x_train))
        for start in range(0, len(permutation), config["batch_size"]):
            batch = permutation[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_loss = float(loss_function(model(x_calibration), y_calibration).item())
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, train_mask, calibration_mask, {"best_epoch": best_epoch, "calibration_loss": best_loss}


def predict_scores(model, scaler, features):
    values = torch.from_numpy(stable_transform(scaler, features))
    with torch.no_grad():
        probabilities = torch.sigmoid(model(values)).numpy()
    return (probabilities[:, 0] * probabilities[:, 1]).astype(np.float32), probabilities.astype(np.float32)


def commits_for_policy(policy, states, predictions, scores, gold_state, threshold=0.5, margin=0.0):
    commits = [None, None, int(states[2])]
    if policy == "B0":
        return [int(state) for state in states]
    if policy == "B1":
        return commits
    if policy == "B3":
        if states[0] == states[1]:
            commits[1] = int(states[1])
        return commits
    if policy == "O1":
        held = None
        for stage in (0, 1):
            if states[stage] == gold_state:
                held = int(states[stage])
            commits[stage] = held
        commits[2] = int(states[2])
        return commits
    values = [fixed_confidence(value) for value in predictions] if policy in ("B2", "B4") else scores
    held = None
    for stage in (0, 1):
        candidate = int(states[stage])
        score = float(values[stage])
        if held is None and score >= threshold:
            held = candidate
        elif held is not None and candidate != held:
            required = threshold + (margin if policy in ("B4", "M0") else 0.0)
            if score >= required:
                held = candidate
        commits[stage] = held
    commits[2] = int(states[2])
    return commits


def path_row(clip_id, order, policy, label, predictions, scores, threshold, margin):
    states = affect_state(predictions).astype(int).tolist()
    gold_state = int(affect_state([label])[0])
    commits = commits_for_policy(policy, states, predictions, scores, gold_state, threshold, margin)
    prefinal = [value for value in commits[:2] if value is not None]
    transitions, previous = 0, None
    for value in commits:
        if value is not None and previous is not None and value != previous:
            transitions += 1
        if value is not None:
            previous = value
    first_commit = next((stage + 1 for stage, value in enumerate(commits) if value is not None), 3)
    final_correct = states[2] == gold_state
    return {
        "clip_id": str(clip_id), "path": order, "policy": policy, "gold_state": gold_state,
        "final_prediction": float(predictions[2]), "final_state": states[2],
        "commits": json.dumps(commits, separators=(",", ":")),
        "prefinal_commits": len(prefinal), "prefinal_errors": sum(value != gold_state for value in prefinal),
        "revisions": transitions,
        "premature": int(any(value != gold_state for value in prefinal) and final_correct),
        "time_to_first": first_commit, "stage2_covered": int(any(value is not None for value in commits[:2])),
        "final_identity": int(commits[2] == states[2]),
    }


def create_records(ids, labels, predictions, orders, score_map):
    records = []
    for index, clip_id in enumerate(ids):
        for order_index, order in enumerate(orders):
            subsets = [canonical_subset(order[:position]) for position in (1, 2, 3)]
            path_predictions = [float(predictions[subset][index]) for subset in subsets]
            path_scores = [float(score_map[(index, order_index, stage)]) for stage in (0, 1)] + [1.0]
            records.append((str(clip_id), order, float(labels[index]), path_predictions, path_scores))
    return records


def simulate(records, policy_parameters):
    rows = []
    for clip_id, order, label, predictions, scores in records:
        for policy in POLICIES:
            parameters = policy_parameters.get(policy, {"threshold": 0.5, "margin": 0.0})
            rows.append(path_row(clip_id, order, policy, label, predictions, scores, parameters["threshold"], parameters["margin"]))
    return rows


def aggregate(rows):
    result = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        if not selected:
            continue
        prefinal_commits = sum(row["prefinal_commits"] for row in selected)
        prefinal_errors = sum(row["prefinal_errors"] for row in selected)
        labels = np.asarray([row["gold_state"] for row in selected], dtype=int)
        final_states = np.asarray([row["final_state"] for row in selected], dtype=int)
        result[policy] = {
            "paths": len(selected),
            "prefinal_commits": prefinal_commits,
            "prefinal_errors": prefinal_errors,
            "prefinal_committed_error_rate": prefinal_errors / prefinal_commits if prefinal_commits else None,
            "committed_revision_rate": sum(row["revisions"] for row in selected) / len(selected),
            "premature_exposure_rate": sum(row["premature"] for row in selected) / len(selected),
            "stage_two_coverage": sum(row["stage2_covered"] for row in selected) / len(selected),
            "time_to_first_commit": sum(row["time_to_first"] for row in selected) / len(selected),
            "final_state_identity": sum(row["final_identity"] for row in selected) / len(selected),
            "final_macro_f1": float(f1_score(labels, final_states, labels=[-1, 0, 1], average="macro", zero_division=0)),
        }
    return result


def add_final_regression_metrics(metrics, labels, final_predictions):
    mae = float(np.mean(np.abs(labels - final_predictions)))
    correlation = float(np.corrcoef(labels, final_predictions)[0, 1])
    for policy in POLICIES:
        metrics[policy]["final_mae"] = mae
        metrics[policy]["final_correlation"] = correlation


def tune_policy(records, policy, threshold_grid, margin_grid, coverage_target):
    candidates = []
    margins = margin_grid if policy in ("B4", "M0") else [0.0]
    for threshold in threshold_grid:
        for margin in margins:
            rows = [path_row(record[0], record[1], policy, record[2], record[3], record[4], threshold, margin) for record in records]
            stats = aggregate(rows)[policy]
            if stats["stage_two_coverage"] >= coverage_target and stats["prefinal_committed_error_rate"] is not None:
                candidates.append((
                    abs(stats["stage_two_coverage"] - coverage_target),
                    stats["prefinal_committed_error_rate"],
                    stats["committed_revision_rate"],
                    -threshold,
                    {"threshold": float(threshold), "margin": float(margin)},
                ))
    if not candidates:
        raise RuntimeError(f"{policy} 没有满足阶段二覆盖率约束的参数")
    selected = min(candidates, key=lambda item: item[:4])
    return selected[:-1], selected[-1]


def policy_components(rows, policy, clip_ids):
    selected = [row for row in rows if row["policy"] == policy and row["clip_id"] in clip_ids]
    commits = sum(row["prefinal_commits"] for row in selected)
    errors = sum(row["prefinal_errors"] for row in selected)
    error_rate = errors / commits if commits else np.nan
    revision_rate = sum(row["revisions"] for row in selected) / len(selected)
    return error_rate, revision_rate


def paired_bootstrap(rows, repetitions, seed):
    clips = sorted({row["clip_id"] for row in rows})
    by_clip = {clip: [row for row in rows if row["clip_id"] == clip] for clip in clips}
    rng = np.random.default_rng(seed)
    error_differences, revision_relative = [], []
    for _ in range(repetitions):
        sampled = rng.choice(clips, size=len(clips), replace=True)
        sample_rows = []
        for sample_index, clip in enumerate(sampled):
            for row in by_clip[clip]:
                copied = dict(row)
                copied["clip_id"] = f"{sample_index}:{clip}"
                sample_rows.append(copied)
        all_sample_ids = {row["clip_id"] for row in sample_rows}
        b4_error, b4_revision = policy_components(sample_rows, "B4", all_sample_ids)
        m0_error, m0_revision = policy_components(sample_rows, "M0", all_sample_ids)
        error_differences.append(b4_error - m0_error)
        revision_relative.append((b4_revision - m0_revision) / b4_revision if b4_revision > 0 else np.nan)
    def interval(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {"error_absolute_reduction_ci95": interval(error_differences), "revision_relative_reduction_ci95": interval(revision_relative)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--condition-dir", required=True)
    arguments = parser.parse_args()
    started = time.time()
    snapshot_path = Path(arguments.parameter_snapshot).resolve()
    output_dir = Path(arguments.condition_dir).resolve()
    if output_dir.exists():
        raise RuntimeError("实验输出目录已存在，禁止覆盖或续跑")
    output_dir.mkdir(parents=True)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_hash = sha256_file(snapshot_path)
    seed = int(snapshot["randomness"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / snapshot["dataset"]["path"]
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != snapshot["dataset"]["sha256"]:
        raise RuntimeError("数据集 SHA-256 与冻结参数不一致")
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    train = dataset[snapshot["dataset"]["development_train_split"]]
    valid = dataset[snapshot["dataset"]["development_evaluation_split"]]
    train_ids = np.asarray(train["id"]).astype(str)
    valid_ids = np.asarray(valid["id"]).astype(str)
    if len(train_ids) != snapshot["dataset"]["expected_train"] or len(valid_ids) != snapshot["dataset"]["expected_valid"]:
        raise RuntimeError("数据集样本数与冻结参数不一致")
    train_labels = np.asarray(train["regression_labels"], dtype=np.float32).reshape(-1)
    valid_labels = np.asarray(valid["regression_labels"], dtype=np.float32).reshape(-1)
    train_pooled = {"T": pool_modality(train["text"]), "A": pool_modality(train["audio"]), "V": pool_modality(train["vision"])}
    valid_pooled = {"T": pool_modality(valid["text"]), "A": pool_modality(valid["audio"]), "V": pool_modality(valid["vision"])}
    del dataset, train, valid
    groups = np.asarray([source_group(value) for value in train_ids])
    base_config = snapshot["models"]["base"]
    train_predictions, valid_predictions = grouped_ridge_predictions(
        train_pooled, valid_pooled, train_labels, groups, base_config["alpha"], base_config["group_folds"], base_config["solver"]
    )
    orders = snapshot["sampling"]["arrival_orders"]
    train_features, train_targets, train_metadata = build_events(train_ids, train_labels, train_predictions, modality_quality(train_pooled), orders)
    valid_features, _, valid_metadata = build_events(valid_ids, valid_labels, valid_predictions, modality_quality(valid_pooled), orders)
    training = snapshot["training"]
    amac_config = dict(snapshot["models"]["amac"])
    amac_config.update(training)
    model, feature_scaler, _, calibration_mask, training_summary = train_amac(train_features, train_targets, train_metadata, amac_config, seed)
    train_scores, train_heads = predict_scores(model, feature_scaler, train_features)
    valid_scores, valid_heads = predict_scores(model, feature_scaler, valid_features)
    train_score_map = {(item[0], item[1], item[2]): float(train_scores[index]) for index, item in enumerate(train_metadata)}
    valid_score_map = {(item[0], item[1], item[2]): float(valid_scores[index]) for index, item in enumerate(valid_metadata)}
    calibration_indices = {item[0] for index, item in enumerate(train_metadata) if calibration_mask[index]}
    calibration_ids = train_ids[sorted(calibration_indices)]
    calibration_labels = train_labels[sorted(calibration_indices)]
    calibration_predictions = {key: values[sorted(calibration_indices)] for key, values in train_predictions.items()}
    remapped_score_map = {}
    index_map = {old: new for new, old in enumerate(sorted(calibration_indices))}
    for (old_index, order_index, stage), score in train_score_map.items():
        if old_index in index_map:
            remapped_score_map[(index_map[old_index], order_index, stage)] = score
    calibration_records = create_records(calibration_ids, calibration_labels, calibration_predictions, orders, remapped_score_map)
    policy_config = snapshot["policy"]
    policy_parameters = {policy: {"threshold": 0.5, "margin": 0.0} for policy in POLICIES}
    tuning = {}
    for policy in ("B2", "B4", "M0"):
        objective, parameters = tune_policy(
            calibration_records, policy, policy_config["threshold_grid"], policy_config["margin_grid"], policy_config["stage_two_coverage_target"]
        )
        policy_parameters[policy] = parameters
        tuning[policy] = {"parameters": parameters, "selection_tuple": list(objective)}
    valid_records = create_records(valid_ids, valid_labels, valid_predictions, orders, valid_score_map)
    rows = simulate(valid_records, policy_parameters)
    policy_metrics = aggregate(rows)
    add_final_regression_metrics(policy_metrics, valid_labels, valid_predictions["TAV"])
    bootstrap = paired_bootstrap(rows, snapshot["statistics"]["bootstrap_repetitions"], seed + 1)
    b4, m0, b0 = policy_metrics["B4"], policy_metrics["M0"], policy_metrics["B0"]
    error_reduction = b4["prefinal_committed_error_rate"] - m0["prefinal_committed_error_rate"]
    revision_reduction = (b4["committed_revision_rate"] - m0["committed_revision_rate"]) / b4["committed_revision_rate"] if b4["committed_revision_rate"] else float("nan")
    comparison = {
        "m0_vs_b4_prefinal_error_absolute_reduction": error_reduction,
        "m0_vs_b4_revision_relative_reduction": revision_reduction,
        **bootstrap,
    }
    gates = snapshot["gates"]
    checks = {
        "problem_eager_revision": b0["committed_revision_rate"] >= gates["eager_revision_min"],
        "problem_eager_premature": b0["premature_exposure_rate"] >= gates["eager_premature_min"],
        "m0_stage_two_coverage": m0["stage_two_coverage"] >= gates["m0_stage_two_coverage_min"],
        "m0_error_point": error_reduction >= gates["m0_error_absolute_reduction_min"],
        "m0_error_interval": bootstrap["error_absolute_reduction_ci95"][0] > gates["bootstrap_lower_bound_min"],
        "m0_revision_point": revision_reduction >= gates["m0_revision_relative_reduction_min"],
        "m0_revision_interval": bootstrap["revision_relative_reduction_ci95"][0] > gates["bootstrap_lower_bound_min"],
        "final_state_identity": all(value["final_state_identity"] == gates["final_state_identity"] for value in policy_metrics.values()),
    }
    decision = {
        "schema": "amac-development-decision-v1", "status": "completed", "run_id": output_dir.name,
        "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"],
        "checks": checks, "prepare_formal_test": all(checks.values()),
        "scope": "Development evidence only; this decision cannot support a final thesis result claim.",
    }
    metrics_document = {
        "schema": "amac-development-metrics-v1", "status": "completed", "run_id": output_dir.name,
        "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"],
        "policy_parameters": policy_parameters, "tuning": tuning, "training": training_summary,
        "policies": policy_metrics, "comparison": comparison,
    }
    csv_fields = ["clip_id", "path", "policy", "gold_state", "final_prediction", "final_state", "commits", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"]
    with (output_dir / "per_path.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "predictions.npz", train_ids=train_ids, valid_ids=valid_ids,
        train_labels=train_labels, valid_labels=valid_labels,
        **{f"train_oof_{key}": value for key, value in train_predictions.items()},
        **{f"valid_{key}": value for key, value in valid_predictions.items()},
        train_amac_heads=train_heads, valid_amac_heads=valid_heads,
    )
    torch.save({
        "state_dict": model.state_dict(), "input_dim": train_features.shape[1],
        "hidden": amac_config["hidden"], "dropout": amac_config["dropout"],
        "feature_scaler_mean": feature_scaler.mean_, "feature_scaler_scale": feature_scaler.scale_,
    }, output_dir / "model.pt")
    write_json(output_dir / "metrics.json", metrics_document)
    write_json(output_dir / "decision.json", decision)
    costs = {
        "schema": "amac-development-costs-v1", "status": "completed", "run_id": output_dir.name,
        "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"],
        "external_api_usd": 0.0, "wall_seconds": time.time() - started,
    }
    write_json(output_dir / "costs.json", costs)
    artifact_names = ["metrics.json", "predictions.npz", "per_path.csv", "model.pt", "costs.json", "decision.json"]
    manifest = {
        "schema": "amac-development-manifest-v1", "status": "completed_unvalidated", "run_id": output_dir.name,
        "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"],
        "dataset": {"name": snapshot["dataset"]["name"], "sha256": dataset_hash, "used_splits": ["train", "valid"], "forbidden_split_indexed": False},
        "counts": {"train": len(train_ids), "valid": len(valid_ids), "arrival_paths": len(orders), "rows": len(rows)},
        "artifacts": {name: sha256_file(output_dir / name) for name in artifact_names},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"run_id": output_dir.name, "prepare_formal_test": decision["prepare_formal_test"], "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
