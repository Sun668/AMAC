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
    parser.add_argument("--run-id", default="competitive_baselines_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    experiment = Path(__file__).resolve().parents[1]
    template = json.loads((experiment / "parameters.template.json").read_text(encoding="utf-8"))
    template["run_id"] = args.run_id
    source_result = root / template["source_result"]
    source_snapshot = root / template["source_snapshot"]
    artifacts = {
        "source_predictions": source_result / "predictions.npz",
        "source_rows": source_result / "per_path.csv",
        "source_metrics": source_result / "metrics.json",
        "source_validator": source_result / "validator.json",
        "source_parameters": source_snapshot / "parameters.json",
        "development_runner": root / "experiments/exp_2026_09_03_amac_train_validation/scripts/run_development.py",
        "runner": experiment / "scripts/run_baselines.py",
        "validator": experiment / "scripts/validate_baselines.py",
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise RuntimeError("冻结失败，缺少文件：" + ", ".join(missing))
    upstream = json.loads(artifacts["source_validator"].read_text(encoding="utf-8"))
    if upstream.get("status") != "passed":
        raise RuntimeError("冻结失败，上游测试未通过独立验证")
    template["frozen_at"] = datetime.now(timezone.utc).isoformat()
    template["artifacts"] = {
        key: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
        for key, path in artifacts.items()
    }
    destination = experiment / "snapshots" / args.run_id
    if destination.exists():
        raise RuntimeError("快照目录已存在，禁止覆盖")
    destination.mkdir(parents=True)
    output = destination / "parameters.json"
    output.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"code": 0, "message": "竞争性基线快照已冻结", "path": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
