#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="affect_contract_replay_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    experiment = Path(__file__).resolve().parents[1]
    config = json.loads((experiment / "parameters.template.json").read_text(encoding="utf-8"))
    config["run_id"] = args.run_id
    files = {
        "replay_input": experiment / "inputs/replay.jsonl",
        "input_manifest": experiment / "inputs/manifest.json",
        "prepare_script": experiment / "scripts/prepare_replay.py",
        "replay_implementation": experiment / "scripts/replay.go",
        "validator": experiment / "scripts/validate_replay.py",
        "contract": root / "internal/affectcontract/contract.go",
        "agent_tool": root / "internal/tools/affect_contract.go",
        "competitive_validator": root / "experiments/exp_2026_09_03_amac_competitive_baselines/results/competitive_baselines_v1/validator.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError("冻结失败，缺少文件：" + ", ".join(missing))
    if json.loads(files["competitive_validator"].read_text(encoding="utf-8")).get("status") != "passed":
        raise RuntimeError("冻结失败，竞争性基线未通过验证")
    config["frozen_at"] = datetime.now(timezone.utc).isoformat()
    config["artifacts"] = {key: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for key, path in files.items()}
    destination = experiment / "snapshots" / args.run_id
    if destination.exists():
        raise RuntimeError("快照目录已存在，禁止覆盖")
    destination.mkdir(parents=True)
    output = destination / "parameters.json"
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"code": 0, "message": "Agent 回放快照已冻结", "path": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

