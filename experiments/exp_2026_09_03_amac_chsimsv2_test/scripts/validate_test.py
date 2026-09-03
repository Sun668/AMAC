#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-7):
    if a is None or b is None: return a is b
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def aggregate(rows):
    commits = sum(row["prefinal_commits"] for row in rows); errors = sum(row["prefinal_errors"] for row in rows)
    return {"paths": len(rows), "prefinal_commits": commits, "prefinal_errors": errors, "prefinal_committed_error_rate": errors / commits if commits else None, "committed_revision_rate": sum(row["revisions"] for row in rows) / len(rows), "premature_exposure_rate": sum(row["premature"] for row in rows) / len(rows), "stage_two_coverage": sum(row["stage2_covered"] for row in rows) / len(rows), "time_to_first_commit": sum(row["time_to_first"] for row in rows) / len(rows), "final_state_identity": sum(row["final_identity"] for row in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    snapshot_path, output = Path(args.parameter_snapshot).resolve(), Path(args.condition_dir).resolve(); root = Path(__file__).resolve().parents[3]; snapshot = json.loads(snapshot_path.read_text()); errors = []
    matrix = load_module("matrix_validator", root / "experiments/exp_2026_09_03_amac_validation_matrix/scripts/validate_matrix.py")
    required = ["manifest.json", "metrics.json", "decision.json", "costs.json", "per_path.csv", "predictions.npz", "models.pt"]
    for name in required:
        if not (output / name).is_file(): errors.append(f"缺少产物: {name}")
    manifest = json.loads((output / "manifest.json").read_text()); metrics = json.loads((output / "metrics.json").read_text()); decision = json.loads((output / "decision.json").read_text()); snapshot_hash = sha256_file(snapshot_path)
    for key in ("runner", "validator", "development_runner", "matrix_runner"):
        if sha256_file(root / snapshot["environment"][f"{key}_path"]) != snapshot["environment"][f"{key}_sha256"]: errors.append(f"{key} 哈希错误")
    for name, document in (("manifest", manifest), ("metrics", metrics), ("decision", decision)):
        if document.get("parameter_snapshot_sha256") != snapshot_hash or document.get("run_id") != output.name: errors.append(f"{name} 身份错误")
    if manifest.get("used_splits") != ["train", "test"] or manifest.get("selection_split_used") is not False: errors.append("测试划分声明错误")
    for name, digest in manifest.get("artifacts", {}).items():
        if sha256_file(output / name) != digest: errors.append(f"产物哈希错误: {name}")
    rows = []
    with (output / "per_path.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in ("seed", "gold_state", "final_state", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"): row[field] = int(row[field])
            row["final_prediction"] = float(row["final_prediction"]); rows.append(row)
    seeds, conditions = snapshot["sampling"]["seeds"], snapshot["policy"]["conditions"]
    expected = len(seeds) * snapshot["dataset"]["expected_evaluation"] * len(snapshot["sampling"]["arrival_orders"]) * len(conditions)
    identities = Counter((row["seed"], row["clip_id"], row["path"], row["condition"]) for row in rows)
    if len(rows) != expected or len(identities) != expected or any(value != 1 for value in identities.values()): errors.append("轨迹数量或身份错误")
    recomputed = {}
    for seed in seeds:
        recomputed[str(seed)] = {}
        for condition in conditions:
            value = aggregate([row for row in rows if row["seed"] == seed and row["condition"] == condition]); recomputed[str(seed)][condition] = value
            for field, observed in value.items():
                if not close(observed, metrics["per_seed"][str(seed)]["conditions"][condition][field]): errors.append(f"指标错误: {seed}.{condition}.{field}")
    with np.load(output / "predictions.npz", allow_pickle=False) as values:
        labels = values["test_labels"]
        if len(labels) != snapshot["dataset"]["expected_evaluation"]: errors.append("测试预测数量错误")
        for subset in ("T", "A", "V", "TA", "TV", "AV", "TAV"):
            observed = values[f"test_{subset}"]
            if not np.isfinite(observed).all(): errors.append(f"{subset} 预测非有限")
            for field, value in matrix.official(observed, labels).items():
                if not close(value, metrics["official_anchor"][subset][field]): errors.append(f"官方指标错误: {subset}.{field}")
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        for comparator in ("B2", "B3", "B4"):
            own, other = recomputed[str(seed)]["H0"], recomputed[str(seed)][comparator]
            effect = {"error_reduction": other["prefinal_committed_error_rate"] - own["prefinal_committed_error_rate"], "revision_relative_reduction": (other["committed_revision_rate"] - own["committed_revision_rate"]) / other["committed_revision_rate"] if other["committed_revision_rate"] else None}
            effect.update(matrix.paired_intervals(seed_rows, "H0", comparator, snapshot["statistics"]["bootstrap_repetitions"], seed * 1000 + sum(map(ord, "H0" + comparator))))
            for field, value in effect.items():
                recorded = metrics["per_seed"][str(seed)]["comparisons"][comparator][field]
                if isinstance(value, list):
                    if any(not close(a, b) for a, b in zip(value, recorded)): errors.append(f"比较区间错误: {seed}.{comparator}.{field}")
                elif not close(value, recorded): errors.append(f"比较效应错误: {seed}.{comparator}.{field}")
    gates = snapshot["gates"]; vals = lambda comparator, field: [metrics["per_seed"][str(seed)]["comparisons"][comparator][field] for seed in seeds]
    claim = {"coverage": all(recomputed[str(seed)]["H0"]["stage_two_coverage"] >= gates["coverage_min"] for seed in seeds), "vs_b4_error": float(np.mean(vals("B4", "error_reduction"))) >= gates["vs_b4_error_reduction_min"] and all(v > 0 for v in vals("B4", "error_reduction")), "vs_b4_revision": float(np.mean(vals("B4", "revision_relative_reduction"))) >= gates["vs_b4_revision_relative_min"] and all(v > 0 for v in vals("B4", "revision_relative_reduction")), "vs_b4_intervals": all(metrics["per_seed"][str(seed)]["comparisons"]["B4"]["error_reduction_ci95"][0] > 0 and metrics["per_seed"][str(seed)]["comparisons"]["B4"]["revision_relative_reduction_ci95"][0] > 0 for seed in seeds), "vs_b2": all(v > 0 for v in vals("B2", "error_reduction")) and all(v > 0 for v in vals("B2", "revision_relative_reduction")), "b3_context": all(recomputed[str(seed)]["B3"]["stage_two_coverage"] < gates["coverage_min"] or (metrics["per_seed"][str(seed)]["comparisons"]["B3"]["error_reduction"] > 0 and metrics["per_seed"][str(seed)]["comparisons"]["B3"]["revision_relative_reduction"] > 0) for seed in seeds)}
    validity = {"five_seeds_complete": len(recomputed) == 5, "final_identity": all(recomputed[str(seed)][condition]["final_state_identity"] == 1.0 for seed in seeds for condition in conditions), "frozen_selected_condition": snapshot["policy"]["selected_condition"] == "H0"}
    if claim != metrics.get("claim_checks") or validity != metrics.get("validity_checks") or decision.get("valid_result") != all(validity.values()) or decision.get("primary_claim_supported") != all(claim.values()) or decision.get("rerun_allowed") is not False: errors.append("决策重算错误")
    report = {"schema": "amac-chsimsv2-test-validator-v1", "status": "failed" if errors else "passed", "run_id": output.name, "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"], "checks": {"hashes": not any("哈希" in item for item in errors), "rows": not any("轨迹" in item for item in errors), "metrics": not any("指标" in item or "比较" in item for item in errors), "official_anchor": not any("官方" in item for item in errors), "decision": not any("决策" in item for item in errors)}, "errors": errors}
    (output / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
