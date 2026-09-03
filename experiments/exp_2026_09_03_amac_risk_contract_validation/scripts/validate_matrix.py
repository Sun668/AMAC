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
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b, tolerance=1e-7):
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def aggregate(rows):
    commits = sum(row["prefinal_commits"] for row in rows); errors = sum(row["prefinal_errors"] for row in rows)
    return {"paths": len(rows), "prefinal_commits": commits, "prefinal_errors": errors, "prefinal_committed_error_rate": errors / commits if commits else None, "committed_revision_rate": sum(row["revisions"] for row in rows) / len(rows), "premature_exposure_rate": sum(row["premature"] for row in rows) / len(rows), "stage_two_coverage": sum(row["stage2_covered"] for row in rows) / len(rows), "time_to_first_commit": sum(row["time_to_first"] for row in rows) / len(rows), "final_state_identity": sum(row["final_identity"] for row in rows) / len(rows)}


def utility(metrics, policy):
    return metrics["prefinal_committed_error_rate"] + policy["revision_weight"] * metrics["committed_revision_rate"] + policy["wait_weight"] * (metrics["time_to_first_commit"] - 1.0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    snapshot_path, output = Path(args.parameter_snapshot).resolve(), Path(args.condition_dir).resolve()
    root = Path(__file__).resolve().parents[3]
    matrix = load_module("prior_validator", root / "experiments/exp_2026_09_03_amac_validation_matrix/scripts/validate_matrix.py")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")); errors = []
    required = ["manifest.json", "metrics.json", "decision.json", "costs.json", "per_path.csv", "frontier.csv", "predictions.npz", "models.pt"]
    for name in required:
        if not (output / name).is_file(): errors.append(f"缺少产物: {name}")
    manifest = json.loads((output / "manifest.json").read_text()); metrics = json.loads((output / "metrics.json").read_text()); decision = json.loads((output / "decision.json").read_text())
    snapshot_hash = sha256_file(snapshot_path)
    if sha256_file(Path(__file__).resolve()) != snapshot["environment"]["validator_sha256"]: errors.append("验证器自身哈希错误")
    runner = root / snapshot["environment"]["runner_path"]
    if sha256_file(runner) != snapshot["environment"]["runner_sha256"]: errors.append("运行器哈希错误")
    for name, document in (("manifest", manifest), ("metrics", metrics), ("decision", decision)):
        if document.get("parameter_snapshot_sha256") != snapshot_hash: errors.append(f"{name} 快照哈希错误")
        if document.get("run_id") != output.name: errors.append(f"{name} run_id 错误")
    if manifest.get("used_splits") != ["train", "valid"] or manifest.get("forbidden_split_indexed") is not False: errors.append("数据划分声明错误")
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
                if not close(observed, metrics["per_seed"][str(seed)]["conditions"][condition].get(field)): errors.append(f"指标错误: {seed}.{condition}.{field}")
    with np.load(output / "predictions.npz", allow_pickle=False) as values:
        if any("test" in key.lower() for key in values.files): errors.append("预测产物包含 test")
        labels = values["evaluation_labels"]
        for subset in ("T", "A", "V", "TA", "TV", "AV", "TAV"):
            observed = values[f"evaluation_{subset}"]
            if not np.isfinite(observed).all(): errors.append(f"{subset} 预测非有限")
            expected_metrics = matrix.official(observed, labels)
            for field, value in expected_metrics.items():
                if not close(value, metrics["official_anchor"][subset][field]): errors.append(f"官方指标错误: {subset}.{field}")
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        for method in ("M0", "H0"):
            for comparator in ("B2", "B3", "B4", "M0" if method == "H0" else "H0"):
                own, other = recomputed[str(seed)][method], recomputed[str(seed)][comparator]
                expected_effect = {"error_reduction": other["prefinal_committed_error_rate"] - own["prefinal_committed_error_rate"], "revision_relative_reduction": (other["committed_revision_rate"] - own["committed_revision_rate"]) / other["committed_revision_rate"] if other["committed_revision_rate"] else None}
                expected_effect.update(matrix.paired_intervals(seed_rows, method, comparator, snapshot["statistics"]["bootstrap_repetitions"], seed * 1000 + sum(map(ord, method + comparator))))
                recorded = metrics["per_seed"][str(seed)]["comparisons"][method][comparator]
                for field, value in expected_effect.items():
                    if isinstance(value, list):
                        if any(not close(a, b) for a, b in zip(value, recorded[field])): errors.append(f"比较区间错误: {seed}.{method}.{comparator}.{field}")
                    elif not close(value, recorded[field]): errors.append(f"比较效应错误: {seed}.{method}.{comparator}.{field}")
    gates = snapshot["gates"]
    def core(method):
        comp = lambda comparator, field: [metrics["per_seed"][str(seed)]["comparisons"][method][comparator][field] for seed in seeds]
        return {"coverage": all(recomputed[str(seed)][method]["stage_two_coverage"] >= gates["coverage_min"] for seed in seeds), "vs_b4_error": float(np.mean(comp("B4", "error_reduction"))) >= gates["vs_b4_error_reduction_min"] and all(v > 0 for v in comp("B4", "error_reduction")), "vs_b4_revision": float(np.mean(comp("B4", "revision_relative_reduction"))) >= gates["vs_b4_revision_relative_min"] and all(v > 0 for v in comp("B4", "revision_relative_reduction")), "vs_b4_intervals": all(metrics["per_seed"][str(seed)]["comparisons"][method]["B4"]["error_reduction_ci95"][0] > 0 and metrics["per_seed"][str(seed)]["comparisons"][method]["B4"]["revision_relative_reduction_ci95"][0] > 0 for seed in seeds), "vs_b2": all(v > 0 for v in comp("B2", "error_reduction")) and all(v > 0 for v in comp("B2", "revision_relative_reduction")), "b3_service_or_dominance": all(recomputed[str(seed)]["B3"]["stage_two_coverage"] < gates["coverage_min"] or (metrics["per_seed"][str(seed)]["comparisons"][method]["B3"]["error_reduction"] > 0 and metrics["per_seed"][str(seed)]["comparisons"][method]["B3"]["revision_relative_reduction"] > 0) for seed in seeds)}
    m0_core, h0_core = core("M0"), core("H0")
    h0_noninferior = all(recomputed[str(seed)]["H0"]["prefinal_committed_error_rate"] <= recomputed[str(seed)]["M0"]["prefinal_committed_error_rate"] + gates["simpler_error_tolerance"] and recomputed[str(seed)]["H0"]["committed_revision_rate"] <= recomputed[str(seed)]["M0"]["committed_revision_rate"] + gates["simpler_revision_tolerance"] and abs(recomputed[str(seed)]["H0"]["stage_two_coverage"] - recomputed[str(seed)]["M0"]["stage_two_coverage"]) <= gates["simpler_coverage_tolerance"] for seed in seeds)
    deltas = [utility(recomputed[str(seed)]["H0"], snapshot["policy"]) - utility(recomputed[str(seed)]["M0"], snapshot["policy"]) for seed in seeds]
    history_material = float(np.mean(deltas)) >= gates["history_utility_improvement_min"] and sum(value > 0 for value in deltas) >= gates["history_required_seeds"]
    selected = "H0" if all(h0_core.values()) and h0_noninferior and not history_material else ("M0" if all(m0_core.values()) else None)
    final_identity = all(recomputed[str(seed)][condition]["final_state_identity"] == gates["final_identity"] for seed in seeds for condition in conditions)
    checks = {"five_seeds_complete": len(recomputed) == 5, "test_isolation": not any("test" in key.lower() for key in np.load(output / "predictions.npz", allow_pickle=False).files), "final_identity": final_identity, "m0_core": m0_core, "h0_core": h0_core, "h0_noninferior": h0_noninferior, "history_material": history_material, "selection_resolved": selected is not None}
    prepare_test = checks["five_seeds_complete"] and checks["test_isolation"] and checks["final_identity"] and checks["selection_resolved"]
    if checks != metrics.get("checks") or checks != decision.get("checks") or decision.get("selected_condition") != selected or decision.get("prepare_one_shot_test") != prepare_test: errors.append("决策重算错误")
    report = {"schema": "amac-risk-contract-validator-v1", "status": "failed" if errors else "passed", "run_id": output.name, "parameter_snapshot_sha256": snapshot_hash, "protocol_version": snapshot["environment"]["protocol_version"], "checks": {"hashes": not any("哈希" in item for item in errors), "rows": not any("轨迹" in item for item in errors), "metrics": not any("指标" in item or "比较" in item for item in errors), "official_anchor": not any("官方" in item for item in errors), "decision": not any("决策" in item for item in errors)}, "errors": errors}
    (output / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
