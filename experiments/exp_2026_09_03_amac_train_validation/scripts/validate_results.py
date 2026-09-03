#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


POLICIES = ("B0", "B1", "B2", "B3", "B4", "M0", "O1")
ORDERS = ("TAV", "TVA", "ATV", "AVT", "VTA", "VAT")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, tolerance=1e-7):
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--condition-dir", required=True)
    arguments = parser.parse_args()
    snapshot_path = Path(arguments.parameter_snapshot).resolve()
    output_dir = Path(arguments.condition_dir).resolve()
    errors = []
    required = ["manifest.json", "metrics.json", "predictions.npz", "per_path.csv", "model.pt", "costs.json", "decision.json"]
    for name in required:
        if not (output_dir / name).is_file():
            errors.append(f"缺少产物: {name}")
    if errors:
        raise SystemExit("; ".join(errors))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    decision = json.loads((output_dir / "decision.json").read_text(encoding="utf-8"))
    snapshot_hash = sha256_file(snapshot_path)
    for document_name, document in (("manifest", manifest), ("metrics", metrics), ("decision", decision)):
        if document.get("parameter_snapshot_sha256") != snapshot_hash:
            errors.append(f"{document_name} 参数快照哈希不一致")
        if document.get("run_id") != output_dir.name:
            errors.append(f"{document_name} run_id 不一致")
        if document.get("protocol_version") != snapshot["environment"]["protocol_version"]:
            errors.append(f"{document_name} 协议版本不一致")
    if manifest.get("dataset", {}).get("used_splits") != ["train", "valid"] or manifest.get("dataset", {}).get("forbidden_split_indexed") is not False:
        errors.append("数据划分声明不符合开发实验协议")
    if manifest.get("dataset", {}).get("sha256") != snapshot["dataset"]["sha256"]:
        errors.append("数据集哈希不一致")
    for name, expected_hash in manifest.get("artifacts", {}).items():
        if not (output_dir / name).is_file() or sha256_file(output_dir / name) != expected_hash:
            errors.append(f"产物哈希不一致: {name}")
    rows = []
    with (output_dir / "per_path.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in ("gold_state", "final_state", "prefinal_commits", "prefinal_errors", "revisions", "premature", "time_to_first", "stage2_covered", "final_identity"):
                row[field] = int(row[field])
            row["final_prediction"] = float(row["final_prediction"])
            row["commits"] = json.loads(row["commits"])
            rows.append(row)
    expected_rows = snapshot["dataset"]["expected_valid"] * len(ORDERS) * len(POLICIES)
    if len(rows) != expected_rows:
        errors.append(f"轨迹行数错误: {len(rows)} != {expected_rows}")
    keys = Counter((row["clip_id"], row["path"], row["policy"]) for row in rows)
    if any(count != 1 for count in keys.values()) or len(keys) != expected_rows:
        errors.append("clip-path-policy 轨迹不唯一或不完整")
    if {row["path"] for row in rows} != set(ORDERS) or {row["policy"] for row in rows} != set(POLICIES):
        errors.append("到达顺序或策略集合不完整")
    per_clip = Counter(row["clip_id"] for row in rows)
    if any(count != len(ORDERS) * len(POLICIES) for count in per_clip.values()):
        errors.append("六条路径未按原始 clip 完整绑定")
    recomputed = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        commits = sum(row["prefinal_commits"] for row in selected)
        prefinal_errors = sum(row["prefinal_errors"] for row in selected)
        labels = [row["gold_state"] for row in selected]
        final_states = [row["final_state"] for row in selected]
        recomputed[policy] = {
            "paths": len(selected), "prefinal_commits": commits, "prefinal_errors": prefinal_errors,
            "prefinal_committed_error_rate": prefinal_errors / commits if commits else None,
            "committed_revision_rate": sum(row["revisions"] for row in selected) / len(selected),
            "premature_exposure_rate": sum(row["premature"] for row in selected) / len(selected),
            "stage_two_coverage": sum(row["stage2_covered"] for row in selected) / len(selected),
            "time_to_first_commit": sum(row["time_to_first"] for row in selected) / len(selected),
            "final_state_identity": sum(row["final_identity"] for row in selected) / len(selected),
            "final_macro_f1": float(f1_score(labels, final_states, labels=[-1, 0, 1], average="macro", zero_division=0)),
        }
    for policy in POLICIES:
        recorded = metrics["policies"][policy]
        for field, value in recomputed[policy].items():
            if not close(value, recorded.get(field)):
                errors.append(f"指标不一致: {policy}.{field}")
    with np.load(output_dir / "predictions.npz", allow_pickle=False) as arrays:
        if len(arrays["valid_ids"]) != snapshot["dataset"]["expected_valid"]:
            errors.append("预测文件 valid 数量错误")
        if any("test" in name.lower() for name in arrays.files):
            errors.append("预测文件包含 test 产物")
        labels = arrays["valid_labels"].astype(np.float32)
        final_predictions = arrays["valid_TAV"].astype(np.float32)
        mae = float(np.mean(np.abs(labels - final_predictions)))
        correlation = float(np.corrcoef(labels, final_predictions)[0, 1])
        if not np.isfinite(final_predictions).all():
            errors.append("预测包含非有限值")
    for policy in POLICIES:
        if not close(metrics["policies"][policy].get("final_mae"), mae):
            errors.append(f"最终 MAE 不一致: {policy}")
        if not close(metrics["policies"][policy].get("final_correlation"), correlation):
            errors.append(f"最终相关系数不一致: {policy}")
    b4 = recomputed["B4"]
    m0 = recomputed["M0"]
    b0 = recomputed["B0"]
    comparison = metrics["comparison"]
    error_reduction = b4["prefinal_committed_error_rate"] - m0["prefinal_committed_error_rate"]
    revision_reduction = (b4["committed_revision_rate"] - m0["committed_revision_rate"]) / b4["committed_revision_rate"] if b4["committed_revision_rate"] else float("nan")
    if not close(error_reduction, comparison["m0_vs_b4_prefinal_error_absolute_reduction"]):
        errors.append("M0/B4 错误率差异不一致")
    if not close(revision_reduction, comparison["m0_vs_b4_revision_relative_reduction"]):
        errors.append("M0/B4 修订率差异不一致")
    gates = snapshot["gates"]
    expected_checks = {
        "problem_eager_revision": b0["committed_revision_rate"] >= gates["eager_revision_min"],
        "problem_eager_premature": b0["premature_exposure_rate"] >= gates["eager_premature_min"],
        "m0_stage_two_coverage": m0["stage_two_coverage"] >= gates["m0_stage_two_coverage_min"],
        "m0_error_point": error_reduction >= gates["m0_error_absolute_reduction_min"],
        "m0_error_interval": comparison["error_absolute_reduction_ci95"][0] > gates["bootstrap_lower_bound_min"],
        "m0_revision_point": revision_reduction >= gates["m0_revision_relative_reduction_min"],
        "m0_revision_interval": comparison["revision_relative_reduction_ci95"][0] > gates["bootstrap_lower_bound_min"],
        "final_state_identity": all(value["final_state_identity"] == gates["final_state_identity"] for value in recomputed.values()),
    }
    if decision.get("checks") != expected_checks or decision.get("prepare_formal_test") != all(expected_checks.values()):
        errors.append("继续/停止决策与冻结门槛不一致")
    report = {
        "schema": "amac-independent-validator-v1", "status": "failed" if errors else "passed",
        "run_id": output_dir.name, "parameter_snapshot_sha256": snapshot_hash,
        "protocol_version": snapshot["environment"]["protocol_version"],
        "checks": {"artifact_hashes": not any("哈希" in error for error in errors), "row_completeness": not any("轨迹" in error or "路径" in error for error in errors), "metric_recomputation": not any("指标" in error or "MAE" in error or "相关" in error or "差异" in error for error in errors), "decision_recomputation": not any("决策" in error for error in errors)},
        "errors": errors,
    }
    (output_dir / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
