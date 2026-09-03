#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ARCHIVED = ("H0", "B2", "B4")
NEW = ("LR", "PLATT", "ISO")
FEATURE_INDICES = (0, 1, 8, 9, 10, 11, 12, 13, 14)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("amac_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_group(value):
    return str(value).split("$_$")[0]


def is_calibration_group(group):
    value = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16)
    return value % 5 == 0


def score_map(metadata, scores):
    return {(item[0], item[1], item[2]): float(scores[index]) for index, item in enumerate(metadata)}


def aggregate(rows):
    commits = sum(int(row["prefinal_commits"]) for row in rows)
    errors = sum(int(row["prefinal_errors"]) for row in rows)
    paths = len(rows)
    return {
        "paths": paths,
        "prefinal_commits": commits,
        "prefinal_errors": errors,
        "prefinal_committed_error_rate": errors / commits if commits else None,
        "committed_revision_rate": sum(int(row["revisions"]) for row in rows) / paths,
        "premature_exposure_rate": sum(int(row["premature"]) for row in rows) / paths,
        "stage_two_coverage": sum(int(row["stage2_covered"]) for row in rows) / paths,
        "time_to_first_commit": sum(int(row["time_to_first"]) for row in rows) / paths,
        "final_state_identity": sum(int(row["final_identity"]) for row in rows) / paths,
    }


def evaluate(base, records, condition, parameters):
    rows = []
    for clip_id, order, label, predictions, scores in records:
        row = base.path_row(
            clip_id, order, "M0", label, predictions, scores,
            parameters["threshold"], parameters["margin"],
        )
        row["condition"] = condition
        row.pop("policy", None)
        rows.append(row)
    return rows


def tune(base, records, condition, config):
    candidates = []
    for threshold in config["threshold_grid"]:
        for margin in config["margin_grid"]:
            parameters = {"threshold": threshold, "margin": margin}
            metrics = aggregate(evaluate(base, records, condition, parameters))
            error = metrics["prefinal_committed_error_rate"]
            if error is None or metrics["stage_two_coverage"] < config["target_coverage"]:
                continue
            utility = (
                error
                + config["revision_weight"] * metrics["committed_revision_rate"]
                + config["wait_weight"] * (metrics["time_to_first_commit"] - 1.0)
            )
            candidates.append((utility, abs(metrics["stage_two_coverage"] - config["target_coverage"]), error, metrics["committed_revision_rate"], -threshold, margin))
    if not candidates:
        raise RuntimeError(f"{condition} 没有满足冻结覆盖率的参数")
    selected = min(candidates)
    return {"threshold": float(-selected[4]), "margin": float(selected[5]), "objective": float(selected[0])}


def utility(metrics, config):
    return float(
        metrics["prefinal_committed_error_rate"]
        + config["revision_weight"] * metrics["committed_revision_rate"]
        + config["wait_weight"] * (metrics["time_to_first_commit"] - 1.0)
    )


def grouped_components(rows, condition):
    selected = [row for row in rows if row["condition"] == condition]
    clips = sorted({row["clip_id"] for row in selected})
    index = {clip: position for position, clip in enumerate(clips)}
    values = np.zeros((len(clips), 5), dtype=np.float64)
    for row in selected:
        current = values[index[row["clip_id"]]]
        current[0] += int(row["prefinal_commits"])
        current[1] += int(row["prefinal_errors"])
        current[2] += int(row["revisions"])
        current[3] += int(row["stage2_covered"])
        current[4] += 1
    return clips, values


def paired_bootstrap(rows, method, comparator, repetitions, seed):
    clips_m, method_values = grouped_components(rows, method)
    clips_c, comparator_values = grouped_components(rows, comparator)
    if clips_m != clips_c:
        raise RuntimeError("配对 bootstrap 的样本身份不一致")
    rng = np.random.default_rng(seed)
    error_reduction, revision_reduction = [], []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(clips_m), len(clips_m))
        m = method_values[sampled].sum(axis=0)
        c = comparator_values[sampled].sum(axis=0)
        error_reduction.append(c[1] / c[0] - m[1] / m[0])
        revision_reduction.append(c[2] / c[4] - m[2] / m[4])
    return {
        "error_absolute_reduction": {
            "estimate": float(comparator_values[:, 1].sum() / comparator_values[:, 0].sum() - method_values[:, 1].sum() / method_values[:, 0].sum()),
            "ci95": [float(value) for value in np.quantile(error_reduction, [0.025, 0.975])],
        },
        "revision_absolute_reduction": {
            "estimate": float(comparator_values[:, 2].sum() / comparator_values[:, 4].sum() - method_values[:, 2].sum() / method_values[:, 4].sum()),
            "ci95": [float(value) for value in np.quantile(revision_reduction, [0.025, 0.975])],
        },
    }


def summarize(per_seed, conditions):
    fields = ("prefinal_committed_error_rate", "committed_revision_rate", "premature_exposure_rate", "stage_two_coverage", "time_to_first_commit")
    output = {}
    for condition in conditions:
        output[condition] = {}
        for field in fields:
            values = [per_seed[str(seed)]["conditions"][condition][field] for seed in sorted(map(int, per_seed))]
            output[condition][field] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--condition-dir", required=True)
    args = parser.parse_args()
    started = time.time()
    snapshot_path = Path(args.parameter_snapshot).resolve()
    output = Path(args.condition_dir).resolve()
    if output.exists():
        raise RuntimeError("输出目录已存在，禁止覆盖")
    root = Path(__file__).resolve().parents[3]
    config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for artifact in config["artifacts"].values():
        path = root / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"冻结文件哈希不一致：{path}")
    output.mkdir(parents=True)
    artifacts = config["artifacts"]
    base = load_module(root / artifacts["development_runner"]["path"])
    predictions = np.load(root / artifacts["source_predictions"]["path"], allow_pickle=False)
    orders = ("TAV", "TVA", "ATV", "AVT", "VTA", "VAT")
    train_ids = predictions["train_ids"].astype(str)
    test_ids = predictions["test_ids"].astype(str)
    train_labels = predictions["train_labels"].astype(np.float32)
    test_labels = predictions["test_labels"].astype(np.float32)
    train_predictions = {subset: predictions[f"train_oof_{subset}"] for subset in base.SUBSETS}
    test_predictions = {subset: predictions[f"test_{subset}"] for subset in base.SUBSETS}
    zero_train_quality = {modality: np.zeros((len(train_ids), 2), dtype=np.float32) for modality in base.MODALITIES}
    zero_test_quality = {modality: np.zeros((len(test_ids), 2), dtype=np.float32) for modality in base.MODALITIES}
    train_features, targets, train_metadata = base.build_events(train_ids, train_labels, train_predictions, zero_train_quality, orders)
    test_features, _, test_metadata = base.build_events(test_ids, test_labels, test_predictions, zero_test_quality, orders)
    x_train = train_features[:, FEATURE_INDICES].astype(np.float64)
    x_test = test_features[:, FEATURE_INDICES].astype(np.float64)
    y_train = targets[:, 0].astype(int)
    groups = np.asarray([item[-1] for item in train_metadata])
    calibration_mask = np.asarray([is_calibration_group(group) for group in groups])
    fit_mask = ~calibration_mask
    if not calibration_mask.any() or not fit_mask.any():
        raise RuntimeError("训练或内部校准分组为空")
    scaler = StandardScaler().fit(x_train[fit_mask])
    lr = LogisticRegression(max_iter=2000, random_state=0).fit(scaler.transform(x_train[fit_mask]), y_train[fit_mask])
    raw_train = np.asarray([base.fixed_confidence(value) for value in train_features[:, 0]], dtype=np.float64)
    raw_test = np.asarray([base.fixed_confidence(value) for value in test_features[:, 0]], dtype=np.float64)
    platt = LogisticRegression(max_iter=2000, random_state=0).fit(raw_train[fit_mask, None], y_train[fit_mask])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_train[fit_mask], y_train[fit_mask])
    train_scores = {
        "LR": lr.predict_proba(scaler.transform(x_train))[:, 1],
        "PLATT": platt.predict_proba(raw_train[:, None])[:, 1],
        "ISO": iso.predict(raw_train),
    }
    test_scores = {
        "LR": lr.predict_proba(scaler.transform(x_test))[:, 1],
        "PLATT": platt.predict_proba(raw_test[:, None])[:, 1],
        "ISO": iso.predict(raw_test),
    }
    calibration_groups = {source_group(value) for value in train_ids if is_calibration_group(source_group(value))}
    archived = pd.read_csv(root / artifacts["source_rows"]["path"])
    archived = archived[archived["condition"].isin(ARCHIVED)].copy()
    all_rows, per_seed, selected_parameters = [], {}, {}
    for seed_index, seed in enumerate(config["seeds"]):
        seed_rows = archived[archived["seed"] == seed].to_dict("records")
        selected_parameters[str(seed)] = {}
        for condition in NEW:
            train_records = base.create_records(train_ids, train_labels, train_predictions, orders, score_map(train_metadata, train_scores[condition]))
            calibration_records = [record for record in train_records if source_group(record[0]) in calibration_groups]
            parameters = tune(base, calibration_records, condition, config)
            selected_parameters[str(seed)][condition] = parameters
            test_records = base.create_records(test_ids, test_labels, test_predictions, orders, score_map(test_metadata, test_scores[condition]))
            seed_rows.extend(evaluate(base, test_records, condition, parameters))
        for row in seed_rows:
            row["seed"] = seed
        all_rows.extend(seed_rows)
        condition_metrics = {condition: aggregate([row for row in seed_rows if row["condition"] == condition]) for condition in config["conditions"]}
        per_seed[str(seed)] = {
            "conditions": condition_metrics,
            "utility": {condition: utility(condition_metrics[condition], config) for condition in config["conditions"]},
        }
    fields = ["seed", "clip_id", "path", "condition", "gold_state", "final_prediction", "final_state", "commits", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"]
    with (output / "per_path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in all_rows])
    comparisons = {}
    for seed in config["seeds"]:
        rows = [row for row in all_rows if int(row["seed"]) == seed]
        comparisons[str(seed)] = {
            f"{method}_vs_{comparator}": paired_bootstrap(rows, method, comparator, config["bootstrap_samples"], config["bootstrap_seed"] + seed)
            for method, comparator in (("LR", "PLATT"), ("LR", "ISO"), ("LR", "B4"), ("H0", "LR"))
        }
    tolerance = config["lr_noninferiority"]
    noninferior = []
    scalar_wins = []
    for seed in config["seeds"]:
        values = per_seed[str(seed)]
        h0, lr_metrics = values["conditions"]["H0"], values["conditions"]["LR"]
        noninferior.append(
            lr_metrics["prefinal_committed_error_rate"] <= h0["prefinal_committed_error_rate"] + tolerance["committed_error_rate"]
            and lr_metrics["committed_revision_rate"] <= h0["committed_revision_rate"] + tolerance["revision_rate"]
            and lr_metrics["stage_two_coverage"] >= h0["stage_two_coverage"] - tolerance["coverage"]
        )
        scalar_wins.append(values["utility"]["LR"] < min(values["utility"]["PLATT"], values["utility"]["ISO"]))
    lr_noninferior = bool(all(noninferior))
    scalar_wins_count = int(sum(scalar_wins))
    all_coverage = bool(all(per_seed[str(seed)]["conditions"][condition]["stage_two_coverage"] >= config["target_coverage"] for seed in config["seeds"] for condition in NEW))
    final_identity = bool(all(per_seed[str(seed)]["conditions"][condition]["final_state_identity"] == 1.0 for seed in config["seeds"] for condition in config["conditions"]))
    decision = {
        "selected_estimator": "LR" if lr_noninferior else "H0",
        "architecture_claim_allowed": not lr_noninferior,
        "feature_information_beyond_scalar_calibration_supported": scalar_wins_count >= config["feature_based_utility_wins_required"],
        "paper_gate_passed": all_coverage and final_identity and scalar_wins_count >= config["feature_based_utility_wins_required"],
        "checks": {
            "all_new_conditions_coverage": all_coverage,
            "final_identity": final_identity,
            "lr_noninferior_to_h0_each_seed": noninferior,
            "lr_utility_beats_both_scalar_calibrators_each_seed": scalar_wins,
            "lr_scalar_win_count": scalar_wins_count,
        },
    }
    metrics = {
        "schema": "amac-competitive-baselines-v1",
        "run_id": config["run_id"],
        "parameter_snapshot_sha256": sha256(snapshot_path),
        "per_seed": per_seed,
        "summary": summarize(per_seed, config["conditions"]),
        "paired_bootstrap": comparisons,
        "selected_parameters": selected_parameters,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "scores.npz", train_lr=train_scores["LR"], train_platt=train_scores["PLATT"], train_iso=train_scores["ISO"], test_lr=test_scores["LR"], test_platt=test_scores["PLATT"], test_iso=test_scores["ISO"])
    manifest = {"parameters.json": sha256(snapshot_path), "metrics.json": sha256(output / "metrics.json"), "decision.json": sha256(output / "decision.json"), "per_path.csv": sha256(output / "per_path.csv"), "scores.npz": sha256(output / "scores.npz")}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"code": 0, "message": "竞争性风险基线运行完成", "decision": decision, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
