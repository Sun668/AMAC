#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY = ("B0", "B1", "B2", "B3", "B4", "H0", "O1")
COMPETITIVE = ("LR", "PLATT", "ISO")
ALL = PRIMARY + COMPETITIVE


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("amac_base_p0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_group(value):
    return str(value).split("$_$", 1)[0]


def load_rows(path, allowed):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in allowed:
                continue
            for field in ("seed", "gold_state", "final_state", "prefinal_commits",
                          "prefinal_errors", "revisions", "premature",
                          "time_to_first", "stage2_covered", "final_identity"):
                row[field] = int(row[field])
            row["final_prediction"] = float(row["final_prediction"])
            row["commits"] = json.loads(row["commits"])
            rows.append(row)
    return rows


def aggregate(rows):
    paths = len(rows)
    commits = sum(row["prefinal_commits"] for row in rows)
    errors = sum(row["prefinal_errors"] for row in rows)
    opportunities = sum(3 - row["time_to_first"] for row in rows)
    revisions = sum(row["revisions"] for row in rows)
    result = {
        "paths": paths,
        "prefinal_commits": commits,
        "prefinal_errors": errors,
        "prefinal_committed_error_rate": errors / commits if commits else None,
        "committed_state_coverage": commits / (2 * paths) if paths else None,
        "committed_revision_rate_per_path": revisions / paths if paths else None,
        "revision_opportunities": opportunities,
        "revision_rate_per_opportunity": revisions / opportunities if opportunities else None,
        "premature_exposure_rate": sum(row["premature"] for row in rows) / paths if paths else None,
        "stage_two_path_coverage": sum(row["stage2_covered"] for row in rows) / paths if paths else None,
        "time_to_first_commit": sum(row["time_to_first"] for row in rows) / paths if paths else None,
        "final_state_identity": sum(row["final_identity"] for row in rows) / paths if paths else None,
    }
    for index, stage in enumerate((1, 2)):
        values = [row["commits"][index] for row in rows]
        count = sum(value is not None for value in values)
        stage_errors = sum(value is not None and value != row["gold_state"] for value, row in zip(values, rows))
        result[f"stage{stage}_committed_count"] = count
        result[f"stage{stage}_coverage"] = count / paths if paths else None
        result[f"stage{stage}_errors"] = stage_errors
        result[f"stage{stage}_committed_error_rate"] = stage_errors / count if count else None
    return result


def summarize(rows, clip_ids=None):
    selected = rows if clip_ids is None else [row for row in rows if row["clip_id"] in clip_ids]
    output = {}
    for condition in ALL:
        condition_rows = [row for row in selected if row["condition"] == condition]
        seeds = sorted({row["seed"] for row in condition_rows})
        per_seed = {str(seed): aggregate([row for row in condition_rows if row["seed"] == seed]) for seed in seeds}
        summary = {}
        for field in next(iter(per_seed.values())):
            values = [per_seed[str(seed)][field] for seed in seeds]
            if any(value is None for value in values):
                summary[field] = {"mean": None, "std": None}
            else:
                summary[field] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
        output[condition] = {"per_seed": per_seed, "summary": summary}
    return output


def grouped(rows, condition, clip_ids=None):
    selected = [row for row in rows if row["condition"] == condition and
                (clip_ids is None or row["clip_id"] in clip_ids)]
    clips = sorted({row["clip_id"] for row in selected})
    index = {clip: position for position, clip in enumerate(clips)}
    values = np.zeros((len(clips), 5), dtype=np.float64)
    for row in selected:
        values[index[row["clip_id"]]] += [
            row["prefinal_commits"], row["prefinal_errors"], row["revisions"],
            3 - row["time_to_first"], 1,
        ]
    return clips, values


def paired_bootstrap(rows, repetitions, seed, clip_ids=None):
    clips_h, h0 = grouped(rows, "H0", clip_ids)
    clips_b, b4 = grouped(rows, "B4", clip_ids)
    if clips_h != clips_b:
        raise RuntimeError("配对 bootstrap 样本身份不一致")
    rng = np.random.default_rng(seed)
    sampled_effects = [[], [], []]
    for _ in range(repetitions):
        sample = rng.integers(0, len(clips_h), len(clips_h))
        h, b = h0[sample].sum(axis=0), b4[sample].sum(axis=0)
        sampled_effects[0].append(b[1] / b[0] - h[1] / h[0])
        sampled_effects[1].append(b[2] / b[4] - h[2] / h[4])
        sampled_effects[2].append(b[2] / b[3] - h[2] / h[3])
    points = [
        b4[:, 1].sum() / b4[:, 0].sum() - h0[:, 1].sum() / h0[:, 0].sum(),
        b4[:, 2].sum() / b4[:, 4].sum() - h0[:, 2].sum() / h0[:, 4].sum(),
        b4[:, 2].sum() / b4[:, 3].sum() - h0[:, 2].sum() / h0[:, 3].sum(),
    ]
    names = ("error_absolute_reduction", "revision_per_path_absolute_reduction",
             "revision_per_opportunity_absolute_reduction")
    return {
        name: {"estimate": float(point), "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])]}
        for name, point, values in zip(names, points, sampled_effects)
    }


def score_map(metadata, values):
    return {(item[0], item[1], item[2]): float(values[index]) for index, item in enumerate(metadata)}


def policy_rows(base, records, condition, threshold, margin):
    policy = condition if condition in ("B2", "B4") else "M0"
    rows = []
    for clip_id, order, label, predictions, scores in records:
        row = base.path_row(clip_id, order, policy, label, predictions, scores, threshold, margin)
        row["condition"] = condition
        row["seed"] = 1
        row["commits"] = json.loads(row["commits"])
        row.pop("policy", None)
        rows.append(row)
    return rows


def curve_summary(points, lower, upper):
    grouped_points = defaultdict(list)
    for point in points:
        if point["prefinal_committed_error_rate"] is not None:
            grouped_points[round(point["committed_state_coverage"], 12)].append(
                point["prefinal_committed_error_rate"])
    x = np.asarray(sorted(grouped_points), dtype=float)
    y = np.asarray([np.mean(grouped_points[value]) for value in sorted(grouped_points)], dtype=float)
    result = {
        "coverage_lower": lower, "coverage_upper": upper,
        "observed_coverage_min": float(x[0]), "observed_coverage_max": float(x[-1]),
    }
    if x[0] > lower or x[-1] < upper:
        result.update({"normalized_aurc": None, "error_at_0_80": None, "error_at_0_90": None})
        return result
    grid = np.linspace(lower, upper, 501)
    result.update({
        "normalized_aurc": float(np.trapezoid(np.interp(grid, x, y), grid) / (upper - lower)),
        "error_at_0_80": float(np.interp(0.80, x, y)),
        "error_at_0_90": float(np.interp(0.90, x, y)),
    })
    return result


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
    root = Path(__file__).resolve().parents[3]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_hash = sha256(snapshot_path)
    for item in snapshot["artifacts"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"冻结输入哈希不一致: {item['path']}")
    for key in ("runner", "validator", "paper_renderer"):
        path = root / snapshot["environment"][f"{key}_path"]
        if sha256(path) != snapshot["environment"][f"{key}_sha256"]:
            raise RuntimeError(f"{key} 哈希不一致")
    output.mkdir(parents=True)
    artifacts = snapshot["artifacts"]
    predictions = np.load(root / artifacts["primary_predictions"]["path"], allow_pickle=False)
    train_ids = predictions["train_ids"].astype(str)
    test_ids = predictions["test_ids"].astype(str)
    train_groups = {source_group(value) for value in train_ids}
    test_groups = {source_group(value) for value in test_ids}
    disjoint_groups = test_groups - train_groups
    disjoint_ids = {value for value in test_ids if source_group(value) in disjoint_groups}
    split_audit = {
        "schema": "amac-source-split-audit-v1",
        "train_samples": len(train_ids), "test_samples": len(test_ids),
        "exact_sample_overlap": len(set(train_ids) & set(test_ids)),
        "train_source_groups": len(train_groups), "test_source_groups": len(test_groups),
        "overlapping_source_groups": len(train_groups & test_groups),
        "source_disjoint_test_groups": len(disjoint_groups),
        "source_disjoint_test_clips": len(disjoint_ids),
        "source_disjoint_test_fraction": len(disjoint_ids) / len(test_ids),
        "source_disjoint_group_ids": sorted(disjoint_groups),
    }
    write_json(output / "source_split_audit.json", split_audit)

    primary_rows = load_rows(root / artifacts["primary_rows"]["path"], PRIMARY)
    primary_rows = [row for row in primary_rows if row["condition"] == "H0" or row["seed"] == 1]
    competitive_rows = load_rows(root / artifacts["competitive_rows"]["path"], COMPETITIVE)
    competitive_rows = [row for row in competitive_rows if row["seed"] == 1]
    all_rows = primary_rows + competitive_rows
    official = summarize(all_rows)
    disjoint = summarize(all_rows, disjoint_ids)
    fields = list(official["H0"]["per_seed"]["1"])
    stage_rows = []
    for scope, summary in (("official_test", official), ("source_disjoint_test", disjoint)):
        for condition in ALL:
            for seed, values in summary[condition]["per_seed"].items():
                stage_rows.append({"scope": scope, "condition": condition, "seed": seed, **values})
    with (output / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "condition", "seed"] + fields)
        writer.writeheader()
        writer.writerows(stage_rows)
    with (output / "current_operating_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "condition"] + fields)
        writer.writeheader()
        for scope, summary in (("official_test", official), ("source_disjoint_test", disjoint)):
            for condition in ALL:
                writer.writerow({"scope": scope, "condition": condition,
                                 **{field: summary[condition]["summary"][field]["mean"] for field in fields}})

    bootstrap = {"official_test": {}, "source_disjoint_test": {}}
    interval_rows = []
    for seed in snapshot["sampling"]["seeds"]:
        seed_rows = [row for row in all_rows if row["condition"] != "H0" or row["seed"] == seed]
        for scope, ids, offset in (("official_test", None, 0),
                                   ("source_disjoint_test", disjoint_ids, 100000)):
            value = paired_bootstrap(
                seed_rows, snapshot["statistics"]["bootstrap_repetitions"],
                snapshot["statistics"]["bootstrap_seed"] + seed + offset, ids)
            bootstrap[scope][str(seed)] = value
            for metric, effect in value.items():
                interval_rows.append({
                    "scope": scope, "seed": seed, "comparison": "H0_vs_B4",
                    "metric": metric, "estimate": effect["estimate"],
                    "ci95_lower": effect["ci95"][0], "ci95_upper": effect["ci95"][1],
                })
    with (output / "bootstrap_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]))
        writer.writeheader()
        writer.writerows(interval_rows)

    base = load_module(root / artifacts["development_runner"]["path"])
    orders = tuple(snapshot["sampling"]["arrival_orders"])
    labels = predictions["test_labels"].astype(np.float32)
    subset_predictions = {subset: predictions[f"test_{subset}"] for subset in base.SUBSETS}
    zero_quality = {modality: np.zeros((len(test_ids), 2), dtype=np.float32)
                    for modality in base.MODALITIES}
    features, _, metadata = base.build_events(test_ids, labels, subset_predictions, zero_quality, orders)
    del features
    scores = np.load(root / artifacts["competitive_scores"]["path"], allow_pickle=False)
    competitive_metrics = json.loads(
        (root / artifacts["competitive_metrics"]["path"]).read_text(encoding="utf-8"))
    primary_metrics = json.loads(
        (root / artifacts["primary_metrics"]["path"]).read_text(encoding="utf-8"))
    primary_parameters = json.loads(
        (root / artifacts["primary_snapshot"]["path"]).read_text(encoding="utf-8")
    )["policy"]["selected_parameters"]
    grid = np.round(np.arange(
        snapshot["policy"]["threshold_grid_start"],
        snapshot["policy"]["threshold_grid_stop"] + snapshot["policy"]["threshold_grid_step"] / 2,
        snapshot["policy"]["threshold_grid_step"]), 10)
    dummy = np.ones(len(metadata), dtype=float)
    specs = [
        ("B2", "1", dummy, primary_parameters["1"]["B2"]["margin"]),
        ("B4", "1", dummy, primary_parameters["1"]["B4"]["margin"]),
    ]
    for index, seed in enumerate(snapshot["sampling"]["seeds"]):
        specs.append(("H0", str(seed), predictions["H0_risk"][index],
                      primary_parameters[str(seed)]["H0"]["margin"]))
    for condition, key in (("LR", "test_lr"), ("PLATT", "test_platt"), ("ISO", "test_iso")):
        specs.append((condition, "1", scores[key],
                      competitive_metrics["selected_parameters"]["1"][condition]["margin"]))
    curve_rows, curves = [], {}
    lower, upper = snapshot["policy"]["aurc_coverage_lower"], snapshot["policy"]["aurc_coverage_upper"]
    for condition, seed, risk, margin in specs:
        records = base.create_records(
            test_ids, labels, subset_predictions, orders, score_map(metadata, risk))
        points = []
        for threshold in grid:
            values = aggregate(policy_rows(base, records, condition, float(threshold), float(margin)))
            point = {"scope": "official_test", "condition": condition, "seed": seed,
                     "threshold": float(threshold), "margin": float(margin), **values}
            curve_rows.append(point)
            points.append(point)
        curves[f"{condition}:{seed}"] = curve_summary(points, lower, upper)
    frontier_fields = ["scope", "condition", "seed", "threshold", "margin"] + fields
    with (output / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=frontier_fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in frontier_fields} for row in curve_rows])

    metrics = {
        "schema": "amac-p0-robustness-metrics-v1", "status": "completed",
        "run_id": output.name, "parameter_snapshot_sha256": snapshot_hash,
        "analysis_status": "post_hoc_supplementary",
        "source_split_audit": split_audit,
        "official_test": official,
        "source_disjoint_test": disjoint,
        "paired_bootstrap_h0_vs_b4": bootstrap,
        "risk_coverage": {
            "definition": "normalized trapezoidal area of prefinal committed error over committed-state coverage",
            "descriptive_only": True, "coverage_interval": [lower, upper], "curves": curves,
        },
        "primary_archived_interval_crosscheck": {
            str(seed): primary_metrics["per_seed"][str(seed)]["comparisons"]["B4"]
            for seed in snapshot["sampling"]["seeds"]
        },
    }
    write_json(output / "metrics.json", metrics)
    validity = {
        "exact_sample_overlap_zero": split_audit["exact_sample_overlap"] == 0,
        "source_disjoint_subset_nonempty": bool(disjoint_ids),
        "source_disjoint_groups_absent_from_train": not (disjoint_groups & train_groups),
        "all_terminal_identities_one": all(row["final_identity"] == 1 for row in all_rows),
        "all_conditions_reported": set(ALL) == set(official),
        "all_frontiers_complete": len(curve_rows) == len(specs) * len(grid),
    }
    source_effects = [
        bootstrap["source_disjoint_test"][str(seed)]["error_absolute_reduction"]
        for seed in snapshot["sampling"]["seeds"]
    ]
    decision = {
        "schema": "amac-p0-robustness-decision-v1",
        "status": "completed_unvalidated", "run_id": output.name,
        "parameter_snapshot_sha256": snapshot_hash,
        "validity_checks": validity, "valid_analysis": all(validity.values()),
        "source_disjoint_error_direction": (
            "favorable_all_seeds" if all(value["estimate"] > 0 for value in source_effects)
            else "mixed_or_reversed"),
        "source_disjoint_error_ci_excludes_zero_all_seeds": all(
            value["ci95"][0] > 0 for value in source_effects),
        "claim_boundary": "Supplementary source-disjoint and descriptive risk-coverage evidence; no confirmatory or non-inferiority claim.",
    }
    write_json(output / "decision.json", decision)
    write_json(output / "costs.json", {
        "schema": "amac-p0-robustness-costs-v1", "status": "completed",
        "run_id": output.name, "external_api_usd": 0.0,
        "wall_seconds": time.time() - started,
    })
    generated = [
        "source_split_audit.json", "current_operating_points.csv", "stage_metrics.csv",
        "risk_coverage.csv", "bootstrap_intervals.csv", "metrics.json",
        "decision.json", "costs.json",
    ]
    write_json(output / "manifest.json", {
        "schema": "amac-p0-robustness-manifest-v1",
        "status": "completed_unvalidated", "run_id": output.name,
        "parameter_snapshot_sha256": snapshot_hash,
        "source_artifacts": snapshot["artifacts"],
        "artifacts": {name: sha256(output / name) for name in generated},
    })
    print(json.dumps({
        "run_id": output.name, "valid_analysis": decision["valid_analysis"],
        "source_disjoint_clips": len(disjoint_ids),
        "source_disjoint_error_direction": decision["source_disjoint_error_direction"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
