#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--template", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    template, output = Path(args.template).resolve(), Path(args.output).resolve()
    if output.exists(): raise RuntimeError("冻结快照已存在，禁止覆盖")
    root = Path(__file__).resolve().parents[3]
    document = json.loads(template.read_text(encoding="utf-8"))
    upstream = document["upstream_validation"]
    decision_path, metrics_path, validator_path = root / upstream["decision_path"], root / upstream["metrics_path"], root / upstream["validator_report_path"]
    decision, metrics, validator = json.loads(decision_path.read_text()), json.loads(metrics_path.read_text()), json.loads(validator_path.read_text())
    if validator.get("status") != "passed" or decision.get("prepare_one_shot_test") is not True or decision.get("selected_condition") != "H0":
        raise RuntimeError("上游验证未授权 H0 一次性测试")
    upstream.update({"decision_sha256": sha256(decision_path), "metrics_sha256": sha256(metrics_path), "validator_report_sha256": sha256(validator_path), "parameter_snapshot_sha256": decision["parameter_snapshot_sha256"]})
    document["policy"]["selected_parameters"] = {seed: {name: values for name, values in metrics["per_seed"][seed]["selected_parameters"].items() if name in ("B2", "B4", "H0")} for seed in metrics["per_seed"]}
    document["status"] = "frozen"
    for key in ("runner", "validator", "development_runner", "matrix_runner"):
        document["environment"][f"{key}_sha256"] = sha256(root / document["environment"][f"{key}_path"])
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__": main()
