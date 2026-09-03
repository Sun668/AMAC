#!/usr/bin/env python3
import argparse, csv, hashlib, importlib.util, json, math
from collections import Counter
from pathlib import Path
import numpy as np

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def close(left, right, tolerance=1e-7):
    if left is None or right is None: return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    snapshot_path = Path(args.parameter_snapshot).resolve(); output = Path(args.condition_dir).resolve(); root = Path(__file__).resolve().parents[3]; config = json.loads(snapshot_path.read_text()); errors = []
    required = ["manifest.json", "metrics.json", "decision.json", "costs.json", "per_path.csv", "predictions.npz", "models.pt"]
    for name in required:
        if not (output / name).is_file(): errors.append(f"缺少产物: {name}")
    manifest = json.loads((output / "manifest.json").read_text()); metrics = json.loads((output / "metrics.json").read_text()); decision = json.loads((output / "decision.json").read_text()); snapshot_hash = sha256(snapshot_path)
    for key in ("runner", "validator", "development_runner", "base_runner", "matrix_runner", "governor_profile"):
        if sha256(root / config["environment"][f"{key}_path"]) != config["environment"][f"{key}_sha256"]: errors.append(f"{key} 哈希错误")
    for document in (manifest, metrics, decision):
        if document.get("run_id") != output.name or document.get("parameter_snapshot_sha256") != snapshot_hash: errors.append("产物身份错误")
    if manifest.get("used_splits") != ["train", "test"] or manifest.get("selection_split_indexed") is not False: errors.append("划分声明错误")
    for name, digest in manifest.get("artifacts", {}).items():
        if sha256(output / name) != digest: errors.append(f"产物哈希错误: {name}")
    rows = []
    with (output / "per_path.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in ("seed", "gold_state", "final_state", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"): row[field] = int(row[field])
            row["final_prediction"] = float(row["final_prediction"]); rows.append(row)
    seeds, conditions = config["sampling"]["risk_seeds"], config["policy"]["reported"]; expected = config["dataset"]["expected_evaluation"] * len(config["sampling"]["arrival_orders"]) * len(seeds) * len(conditions); identities = Counter((row["seed"], row["clip_id"], row["path"], row["policy"]) for row in rows)
    if len(rows) != expected or len(identities) != expected or any(value != 1 for value in identities.values()): errors.append("轨迹数量或身份错误")
    dev = load_module("dev_validator", root / config["environment"]["development_runner_path"]); base = load_module("base_validator", root / config["environment"]["base_runner_path"]); matrix = load_module("matrix_validator", root / config["environment"]["matrix_runner_path"])
    recomputed = {}
    for seed in seeds:
        recomputed[str(seed)] = {}
        for condition in conditions:
            observed = dev.aggregate([row for row in rows if row["seed"] == seed and row["policy"] == condition]); recomputed[str(seed)][condition] = observed; recorded = metrics["per_seed"][str(seed)]["policies"][condition]
            for field, value in observed.items():
                if not close(value, recorded[field]): errors.append(f"策略指标错误: {seed}.{condition}.{field}")
        comparison = metrics["per_seed"][str(seed)]["comparison"]; expected_error = recomputed[str(seed)]["B4"]["prefinal_committed_error_rate"] - recomputed[str(seed)]["H0"]["prefinal_committed_error_rate"]; expected_gap = abs(recomputed[str(seed)]["B4"]["stage_two_coverage"] - recomputed[str(seed)]["H0"]["stage_two_coverage"])
        if not close(expected_error, comparison["error_reduction"]) or not close(expected_gap, comparison["coverage_gap"]): errors.append(f"比较效应错误: {seed}")
        bootstrap_rows = [dict(row, policy="M0" if row["policy"] == "H0" else row["policy"]) for row in rows if row["seed"] == seed and row["policy"] in ("B4", "H0")]; intervals = base.paired_bootstrap(bootstrap_rows, config["statistics"]["bootstrap_repetitions"], seed * 1009)
        for field, values in intervals.items():
            if any(not close(a, b) for a, b in zip(values, comparison[field])): errors.append(f"置信区间错误: {seed}.{field}")
    with np.load(output / "predictions.npz", allow_pickle=False) as values:
        train_ids = values["train_ids"].astype(str); test_ids = values["test_ids"].astype(str); test_labels = values["test_labels"]
        if set(train_ids) & set(test_ids): errors.append("训练与测试 clip 交叉")
        if len(values["source_disjoint_ids"]) != metrics["source_disjoint_clip_count"]: errors.append("source-disjoint 数量错误")
        for subset in config["sampling"]["subsets"]:
            for prefix in ("train_oof", "test"):
                if not np.isfinite(values[f"{prefix}_{subset}"]).all(): errors.append(f"非有限预测: {prefix}_{subset}")
            official = matrix.official_metrics(values[f"test_{subset}"], test_labels)
            for field, value in official.items():
                if not close(value, metrics["terminal"]["masked_fusion"][subset][field]): errors.append(f"终端指标错误: {subset}.{field}")
    gates = config["gates"]; performance = {"terminal_mae": metrics["terminal"]["masked_fusion"]["TAV"]["MAE"] - metrics["terminal"]["ridge_reference"]["TAV"]["MAE"] <= gates["terminal_mae_degradation_max"], "coverage": all(recomputed[str(seed)]["H0"]["stage_two_coverage"] >= gates["h0_coverage_min"] for seed in seeds), "matched_coverage": all(metrics["per_seed"][str(seed)]["comparison"]["coverage_gap"] <= gates["coverage_gap_max"] for seed in seeds), "error_effect": all(metrics["per_seed"][str(seed)]["comparison"]["error_reduction"] >= gates["vs_b4_error_reduction_min"] for seed in seeds), "error_intervals": all(metrics["per_seed"][str(seed)]["comparison"]["error_absolute_reduction_ci95"][0] > 0 for seed in seeds)}
    validity = {"train_test_disjoint": not bool(set(train_ids) & set(test_ids)), "finite_predictions": not any("非有限" in error for error in errors), "five_seeds": len(recomputed) == 5, "final_identity": all(recomputed[str(seed)][condition]["final_state_identity"] == gates["final_identity"] for seed in seeds for condition in conditions), "source_disjoint_nonempty": metrics["source_disjoint_clip_count"] > 0}
    if performance != metrics.get("performance_checks") or validity != metrics.get("validity_checks"): errors.append("门槛重算错误")
    if decision.get("valid_result") != all(validity.values()) or decision.get("primary_claim_supported") != (all(validity.values()) and all(performance.values())) or decision.get("revision_claim") != "exploratory_only" or decision.get("rerun_allowed") is not False: errors.append("决策重算错误")
    report = {"schema": "amac-masked-fusion-test-validator-v1", "status": "failed" if errors else "passed", "run_id": output.name, "parameter_snapshot_sha256": snapshot_hash, "protocol_version": config["environment"]["protocol_version"], "checks": {"hashes": not any("哈希" in error for error in errors), "splits": not any("划分" in error or "交叉" in error for error in errors), "rows": not any("轨迹" in error for error in errors), "metrics": not any("指标" in error or "效应" in error or "区间" in error or "门槛" in error for error in errors), "decision": not any("决策" in error for error in errors)}, "errors": errors}
    (output / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()

