#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-9):
    if a in (None, "") or b in (None, ""):
        return a in (None, "") and b in (None, "")
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def load_source_rows(path, allowed):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in allowed:
                continue
            for field in ("seed", "gold_state", "prefinal_commits", "prefinal_errors",
                          "revisions", "premature", "time_to_first",
                          "stage2_covered", "final_identity"):
                row[field] = int(row[field])
            row["commits"] = json.loads(row["commits"])
            rows.append(row)
    return rows


def aggregate_source(rows):
    paths = len(rows)
    commits = sum(row["prefinal_commits"] for row in rows)
    errors = sum(row["prefinal_errors"] for row in rows)
    revisions = sum(row["revisions"] for row in rows)
    opportunities = sum(3 - row["time_to_first"] for row in rows)
    result = {
        "paths": paths, "prefinal_commits": commits, "prefinal_errors": errors,
        "prefinal_committed_error_rate": errors / commits if commits else None,
        "committed_state_coverage": commits / (2 * paths),
        "committed_revision_rate_per_path": revisions / paths,
        "revision_opportunities": opportunities,
        "revision_rate_per_opportunity": revisions / opportunities if opportunities else None,
        "premature_exposure_rate": sum(row["premature"] for row in rows) / paths,
        "stage_two_path_coverage": sum(row["stage2_covered"] for row in rows) / paths,
        "time_to_first_commit": sum(row["time_to_first"] for row in rows) / paths,
        "final_state_identity": sum(row["final_identity"] for row in rows) / paths,
    }
    for index, stage in enumerate((1, 2)):
        count = sum(row["commits"][index] is not None for row in rows)
        stage_errors = sum(
            row["commits"][index] is not None
            and row["commits"][index] != row["gold_state"] for row in rows)
        result[f"stage{stage}_committed_count"] = count
        result[f"stage{stage}_coverage"] = count / paths
        result[f"stage{stage}_errors"] = stage_errors
        result[f"stage{stage}_committed_error_rate"] = (
            stage_errors / count if count else None)
    return result


def bootstrap_source(rows, repetitions, seed):
    def components(condition):
        selected = [row for row in rows if row["condition"] == condition]
        clips = sorted({row["clip_id"] for row in selected})
        index = {clip: position for position, clip in enumerate(clips)}
        values = np.zeros((len(clips), 5), dtype=float)
        for row in selected:
            values[index[row["clip_id"]]] += [
                row["prefinal_commits"], row["prefinal_errors"],
                row["revisions"], 3 - row["time_to_first"], 1]
        return clips, values
    clips_h, h0 = components("H0")
    clips_b, b4 = components("B4")
    if clips_h != clips_b:
        raise RuntimeError("验证器发现 bootstrap 身份不一致")
    rng = np.random.default_rng(seed)
    sampled = [[], [], []]
    for _ in range(repetitions):
        take = rng.integers(0, len(clips_h), len(clips_h))
        h, b = h0[take].sum(axis=0), b4[take].sum(axis=0)
        sampled[0].append(b[1] / b[0] - h[1] / h[0])
        sampled[1].append(b[2] / b[4] - h[2] / h[4])
        sampled[2].append(b[2] / b[3] - h[2] / h[3])
    points = [
        b4[:, 1].sum() / b4[:, 0].sum() - h0[:, 1].sum() / h0[:, 0].sum(),
        b4[:, 2].sum() / b4[:, 4].sum() - h0[:, 2].sum() / h0[:, 4].sum(),
        b4[:, 2].sum() / b4[:, 3].sum() - h0[:, 2].sum() / h0[:, 3].sum(),
    ]
    names = ("error_absolute_reduction", "revision_per_path_absolute_reduction",
             "revision_per_opportunity_absolute_reduction")
    return {
        name: {"estimate": point, "ci95": np.quantile(values, [0.025, 0.975])}
        for name, point, values in zip(names, points, sampled)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--condition-dir", required=True)
    args = parser.parse_args()
    snapshot_path = Path(args.parameter_snapshot).resolve()
    output = Path(args.condition_dir).resolve()
    root = Path(__file__).resolve().parents[3]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    errors = []
    for item in snapshot["artifacts"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            errors.append(f"冻结输入哈希不一致: {item['path']}")
    for key in ("runner", "validator", "paper_renderer"):
        path = root / snapshot["environment"][f"{key}_path"]
        if sha256(path) != snapshot["environment"][f"{key}_sha256"]:
            errors.append(f"{key} 哈希不一致")
    required = [
        "source_split_audit.json", "current_operating_points.csv",
        "stage_metrics.csv", "risk_coverage.csv", "bootstrap_intervals.csv",
        "metrics.json", "decision.json", "costs.json", "manifest.json",
    ]
    for name in required:
        if not (output / name).is_file():
            errors.append(f"缺少产物: {name}")
    if errors:
        raise RuntimeError("；".join(errors))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    split = json.loads((output / "source_split_audit.json").read_text(encoding="utf-8"))
    snapshot_hash = sha256(snapshot_path)
    for document, name in ((manifest, "manifest"), (metrics, "metrics"), (decision, "decision")):
        if document.get("run_id") != output.name or document.get("parameter_snapshot_sha256") != snapshot_hash:
            errors.append(f"{name} 身份不一致")
    for name, digest in manifest["artifacts"].items():
        if sha256(output / name) != digest:
            errors.append(f"产物哈希不一致: {name}")
    predictions = np.load(
        root / snapshot["artifacts"]["primary_predictions"]["path"],
        allow_pickle=False)
    train_ids = predictions["train_ids"].astype(str)
    test_ids = predictions["test_ids"].astype(str)
    group = lambda value: str(value).split("$_$", 1)[0]
    train_groups = {group(value) for value in train_ids}
    test_groups = {group(value) for value in test_ids}
    disjoint_groups = test_groups - train_groups
    expected = {
        "train_samples": len(train_ids),
        "test_samples": len(test_ids),
        "exact_sample_overlap": len(set(train_ids) & set(test_ids)),
        "train_source_groups": len(train_groups),
        "test_source_groups": len(test_groups),
        "overlapping_source_groups": len(train_groups & test_groups),
        "source_disjoint_test_groups": len(disjoint_groups),
        "source_disjoint_test_clips": sum(group(value) in disjoint_groups for value in test_ids),
    }
    for key, value in expected.items():
        if split.get(key) != value:
            errors.append(f"来源划分审计错误: {key}")
    if set(split["source_disjoint_group_ids"]) & train_groups:
        errors.append("来源隔离组仍出现在训练集")
    primary = load_source_rows(
        root / snapshot["artifacts"]["primary_rows"]["path"],
        {"B0", "B1", "B2", "B3", "B4", "H0", "O1"})
    primary = [row for row in primary if row["condition"] == "H0" or row["seed"] == 1]
    competitive = load_source_rows(
        root / snapshot["artifacts"]["competitive_rows"]["path"],
        {"LR", "PLATT", "ISO"})
    all_rows = primary + [row for row in competitive if row["seed"] == 1]
    disjoint_ids = {value for value in test_ids if group(value) in disjoint_groups}
    with (output / "stage_metrics.csv").open(newline="", encoding="utf-8") as handle:
        stages = list(csv.DictReader(handle))
    if len(stages) != 28:
        errors.append("分阶段指标行数错误")
    if any(not close(row["final_state_identity"], 1.0) for row in stages):
        errors.append("终态身份未保持")
    for row in stages:
        paths = float(row["paths"])
        commits = float(row["prefinal_commits"])
        opportunities = float(row["revision_opportunities"])
        if not close(row["committed_state_coverage"], commits / (2 * paths)):
            errors.append(f"承诺状态覆盖率错误: {row['scope']}.{row['condition']}.{row['seed']}")
        if float(row["stage1_committed_count"]) + float(row["stage2_committed_count"]) != commits:
            errors.append(f"阶段承诺数不守恒: {row['scope']}.{row['condition']}.{row['seed']}")
        if opportunities and not 0 <= float(row["revision_rate_per_opportunity"]) <= 1:
            errors.append(f"修订机会率越界: {row['scope']}.{row['condition']}.{row['seed']}")
        eligible = [
            item for item in all_rows
            if item["condition"] == row["condition"]
            and item["seed"] == int(row["seed"])
            and (row["scope"] == "official_test" or item["clip_id"] in disjoint_ids)]
        recomputed = aggregate_source(eligible)
        for field, value in recomputed.items():
            if not close(row[field], value):
                errors.append(
                    f"来源逐行重算错误: {row['scope']}.{row['condition']}.{row['seed']}.{field}")
    for scope, ids, offset in (("official_test", None, 0),
                               ("source_disjoint_test", disjoint_ids, 100000)):
        for seed in snapshot["sampling"]["seeds"]:
            eligible = [
                row for row in all_rows
                if (row["condition"] != "H0" or row["seed"] == seed)
                and (ids is None or row["clip_id"] in ids)]
            recalculated = bootstrap_source(
                eligible, snapshot["statistics"]["bootstrap_repetitions"],
                snapshot["statistics"]["bootstrap_seed"] + seed + offset)
            recorded = metrics["paired_bootstrap_h0_vs_b4"][scope][str(seed)]
            for metric, value in recalculated.items():
                if not close(value["estimate"], recorded[metric]["estimate"]):
                    errors.append(f"bootstrap 点估计重算错误: {scope}.{seed}.{metric}")
                if any(not close(a, b) for a, b in zip(value["ci95"], recorded[metric]["ci95"])):
                    errors.append(f"bootstrap 区间重算错误: {scope}.{seed}.{metric}")
    with (output / "risk_coverage.csv").open(newline="", encoding="utf-8") as handle:
        frontier = list(csv.DictReader(handle))
    thresholds = int(round(
        (snapshot["policy"]["threshold_grid_stop"] - snapshot["policy"]["threshold_grid_start"])
        / snapshot["policy"]["threshold_grid_step"])) + 1
    curves = 2 + len(snapshot["sampling"]["seeds"]) + 3
    if len(frontier) != thresholds * curves:
        errors.append("风险覆盖曲线行数错误")
    grouped = {}
    for row in frontier:
        key = f"{row['condition']}:{row['seed']}"
        grouped.setdefault(key, []).append(row)
        for field in ("committed_state_coverage", "prefinal_committed_error_rate",
                      "stage_two_path_coverage", "final_state_identity"):
            if row[field] and not 0 <= float(row[field]) <= 1:
                errors.append(f"风险覆盖值越界: {key}.{field}")
    lower, upper = metrics["risk_coverage"]["coverage_interval"]
    for key, rows in grouped.items():
        points = {}
        for row in rows:
            if row["prefinal_committed_error_rate"]:
                coverage = round(float(row["committed_state_coverage"]), 12)
                points.setdefault(coverage, []).append(float(row["prefinal_committed_error_rate"]))
        x = np.asarray(sorted(points))
        y = np.asarray([np.mean(points[value]) for value in sorted(points)])
        if x[0] > lower or x[-1] < upper:
            calculated = None
        else:
            grid = np.linspace(lower, upper, 501)
            calculated = float(np.trapezoid(np.interp(grid, x, y), grid) / (upper - lower))
        if not close(calculated, metrics["risk_coverage"]["curves"][key]["normalized_aurc"]):
            errors.append(f"AURC 重算错误: {key}")
    accepted = not errors and decision.get("valid_analysis") is True
    validator = {
        "schema": "amac-p0-robustness-validator-v1",
        "status": "accepted" if accepted else "rejected",
        "run_id": output.name,
        "parameter_snapshot_sha256": snapshot_hash,
        "accepted": accepted,
        "errors": errors,
        "checks": {
            "source_artifact_hashes": not any("冻结输入" in error for error in errors),
            "sample_and_source_split": not any("来源" in error for error in errors),
            "stage_denominators": not any(
                token in error for error in errors
                for token in ("阶段", "承诺状态", "修订机会")),
            "risk_coverage": not any(
                token in error for error in errors
                for token in ("风险覆盖", "AURC")),
            "terminal_identity": not any("终态" in error for error in errors),
        },
    }
    (output / "validator.json").write_text(
        json.dumps(validator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if not accepted:
        raise RuntimeError("独立验证失败: " + "；".join(errors))
    print(json.dumps({"run_id": output.name, "accepted": True,
                      "checks": validator["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
