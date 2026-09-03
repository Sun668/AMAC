#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, tolerance=1e-12):
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


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


def utility(metrics, config):
    return metrics["prefinal_committed_error_rate"] + config["revision_weight"] * metrics["committed_revision_rate"] + config["wait_weight"] * (metrics["time_to_first_commit"] - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--parameter-snapshot", required=True)
    args = parser.parse_args()
    result = Path(args.result_dir).resolve()
    snapshot_path = Path(args.parameter_snapshot).resolve()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    decision = json.loads((result / "decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    for name in ("metrics.json", "decision.json", "per_path.csv", "scores.npz"):
        if manifest.get(name) != sha256(result / name):
            errors.append(f"产物哈希不匹配：{name}")
    if manifest.get("parameters.json") != sha256(snapshot_path):
        errors.append("参数快照哈希不匹配")
    for artifact in config["artifacts"].values():
        if sha256(root / artifact["path"]) != artifact["sha256"]:
            errors.append(f"上游或代码哈希不匹配：{artifact['path']}")
    with (result / "per_path.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(config["seeds"]) * 1034 * 6 * len(config["conditions"])
    if len(rows) != expected_rows:
        errors.append(f"逐路径行数错误：{len(rows)} != {expected_rows}")
    recomputed = {}
    for seed in config["seeds"]:
        recomputed[str(seed)] = {}
        for condition in config["conditions"]:
            selected = [row for row in rows if int(row["seed"]) == seed and row["condition"] == condition]
            if len(selected) != 1034 * 6:
                errors.append(f"{seed}/{condition} 路径数错误：{len(selected)}")
                continue
            values = aggregate(selected)
            recomputed[str(seed)][condition] = values
            archived = metrics["per_seed"][str(seed)]["conditions"][condition]
            for key, value in values.items():
                if value is None or not close(value, archived[key]):
                    errors.append(f"{seed}/{condition}/{key} 汇总不一致")
    tolerance = config["lr_noninferiority"]
    noninferior, scalar_wins = [], []
    for seed in config["seeds"]:
        conditions = recomputed[str(seed)]
        h0, lr = conditions["H0"], conditions["LR"]
        noninferior.append(
            lr["prefinal_committed_error_rate"] <= h0["prefinal_committed_error_rate"] + tolerance["committed_error_rate"]
            and lr["committed_revision_rate"] <= h0["committed_revision_rate"] + tolerance["revision_rate"]
            and lr["stage_two_coverage"] >= h0["stage_two_coverage"] - tolerance["coverage"]
        )
        scalar_wins.append(utility(lr, config) < min(utility(conditions["PLATT"], config), utility(conditions["ISO"], config)))
    all_coverage = all(recomputed[str(seed)][condition]["stage_two_coverage"] >= config["target_coverage"] for seed in config["seeds"] for condition in ("LR", "PLATT", "ISO"))
    final_identity = all(recomputed[str(seed)][condition]["final_state_identity"] == 1.0 for seed in config["seeds"] for condition in config["conditions"])
    expected_decision = {
        "selected_estimator": "LR" if all(noninferior) else "H0",
        "architecture_claim_allowed": not all(noninferior),
        "feature_information_beyond_scalar_calibration_supported": sum(scalar_wins) >= config["feature_based_utility_wins_required"],
        "paper_gate_passed": all_coverage and final_identity and sum(scalar_wins) >= config["feature_based_utility_wins_required"],
        "checks": {
            "all_new_conditions_coverage": all_coverage,
            "final_identity": final_identity,
            "lr_noninferior_to_h0_each_seed": noninferior,
            "lr_utility_beats_both_scalar_calibrators_each_seed": scalar_wins,
            "lr_scalar_win_count": sum(scalar_wins),
        },
    }
    if decision != expected_decision or metrics["decision"] != expected_decision:
        errors.append("决策与独立重算不一致")
    if config.get("test_labels_used_for_fit_or_tuning") is not False:
        errors.append("协议未明确禁止使用测试标签拟合或调参")
    report = {
        "schema": "amac-competitive-baselines-validator-v1",
        "status": "passed" if not errors else "failed",
        "run_id": config["run_id"],
        "parameter_snapshot_sha256": sha256(snapshot_path),
        "checks": {
            "artifact_hashes": not any("哈希" in error for error in errors),
            "row_identity": not any("路径" in error or "行数" in error for error in errors),
            "metric_recomputation": not any("汇总" in error for error in errors),
            "decision_recomputation": "决策与独立重算不一致" not in errors,
            "test_isolation_declared": config.get("test_labels_used_for_fit_or_tuning") is False,
        },
        "errors": errors,
    }
    (result / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {"status": report["status"], "decision": expected_decision, "summary": metrics["summary"], "validator": report["checks"]}
    (result / "result.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"code": 0 if not errors else 1, "message": "竞争性基线验证通过" if not errors else "竞争性基线验证失败", "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
