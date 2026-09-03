#!/usr/bin/env python3
import argparse
import hashlib
import json
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


def simulate(record):
    held = None
    commits = []
    for stage, observation in enumerate(record["observations"]):
        candidate = observation["provisional_state"]
        score = observation["correctness_probability"]
        if stage == 2:
            held = candidate
        elif held is None and score >= record["threshold"]:
            held = candidate
        elif held is not None and candidate != held and score >= record["threshold"] + record["margin"]:
            held = candidate
        commits.append(held)
    return commits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    snapshot_path = Path(args.parameter_snapshot).resolve()
    result_path = Path(args.result).resolve()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    errors = []
    for artifact in config["artifacts"].values():
        path = root / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            errors.append(f"冻结文件哈希不一致：{artifact['path']}")
    replay_path = root / config["artifacts"]["replay_input"]["path"]
    input_paths = 0
    input_observations = 0
    expected_mismatches = 0
    with replay_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            input_paths += 1
            input_observations += len(record["observations"])
            if simulate(record) != record["expected_commits"]:
                expected_mismatches += 1
    if input_paths != config["expected_paths"] or result.get("paths") != input_paths:
        errors.append("回放路径数不一致")
    if input_observations != config["expected_observations"] or result.get("observations") != input_observations:
        errors.append("观察调用数不一致")
    if result.get("input_sha256") != sha256(replay_path):
        errors.append("结果引用的输入哈希不一致")
    if expected_mismatches != 0:
        errors.append(f"输入中的离线轨迹与独立模拟不一致：{expected_mismatches}")
    if result.get("trajectory_mismatch_count") != 0:
        errors.append(f"在线与离线轨迹不一致：{result.get('trajectory_mismatch_count')}")
    if result.get("final_identity") is not True:
        errors.append("第三模态未保持最终状态身份")
    checks = result.get("robustness_checks", {})
    if not checks or not all(checks.values()) or result.get("robustness_all_passed") is not True:
        errors.append("Agent 鲁棒性检查未全部通过")
    latency = result.get("latency_microseconds", {})
    if not all(isinstance(latency.get(key), (int, float)) and latency[key] >= 0 for key in ("p50", "p95", "p99", "max")):
        errors.append("延迟统计缺失或非法")
    report = {
        "schema": "affect-contract-agent-replay-validator-v1",
        "status": "passed" if not errors else "failed",
        "run_id": config["run_id"],
        "parameter_snapshot_sha256": sha256(snapshot_path),
        "checks": {
            "artifact_hashes": not any("哈希" in error for error in errors),
            "input_trajectory_recomputed": expected_mismatches == 0,
            "online_offline_identity": result.get("trajectory_mismatch_count") == 0,
            "final_identity": result.get("final_identity") is True,
            "robustness": bool(checks) and all(checks.values()),
            "latency_present": bool(latency),
        },
        "errors": errors,
    }
    destination = result_path.parent / "validator.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {"status": report["status"], "paths": input_paths, "observations": input_observations, "trajectory_mismatch_count": result.get("trajectory_mismatch_count"), "final_identity": result.get("final_identity"), "robustness_checks": checks, "latency_microseconds": latency, "validator": report["checks"]}
    (result_path.parent / "result.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"code": 0 if not errors else 1, "message": "Agent 回放验证通过" if not errors else "Agent 回放验证失败", "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
