#!/usr/bin/env python3
import argparse, csv, hashlib, json, math
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def close(a, b): return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
def group(clip_id): return str(clip_id).split("/", 1)[0]

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    snapshot = Path(args.parameter_snapshot).resolve(); output = Path(args.condition_dir).resolve(); root = Path(__file__).resolve().parents[3]; config = json.loads(snapshot.read_text()); errors = []
    for key in ("runner", "validator"):
        if sha256(root / config["environment"][f"{key}_path"]) != config["environment"][f"{key}_sha256"]: errors.append(f"{key} 哈希错误")
    for key in ("rows", "validator", "snapshot"):
        if sha256(root / config["models"][key]) != config["models"][f"{key}_sha256"]: errors.append(f"source.{key} 哈希错误")
    recorded = {}
    with (output / "group_effects.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle): recorded[(int(row["seed"]), row["group"])] = row
    source = []
    with (root / config["models"]["rows"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] in ("B4", "H0"): source.append(row)
    for seed in config["sampling"]["risk_seeds"]:
        for source_group in config["dataset"]["expected_test_groups"]:
            selected = [row for row in source if int(row["seed"]) == seed and group(row["clip_id"]) == source_group]; values = {}
            for condition in ("B4", "H0"):
                subset = [row for row in selected if row["condition"] == condition]
                values[condition] = {"error": sum(int(row["prefinal_errors"]) for row in subset) / sum(int(row["prefinal_commits"]) for row in subset),
                                     "revision": sum(int(row["revisions"]) for row in subset) / len(subset),
                                     "coverage": sum(int(row["stage2_covered"]) for row in subset) / len(subset)}
            row = recorded[(seed, source_group)]; expected = {"error_reduction": values["B4"]["error"] - values["H0"]["error"],
                                                               "revision_path_reduction": values["B4"]["revision"] - values["H0"]["revision"],
                                                               "coverage_gap": abs(values["B4"]["coverage"] - values["H0"]["coverage"])}
            if int(row["clips"]) != len({item["clip_id"] for item in selected if item["condition"] == "B4"}): errors.append(f"组样本数错误:{seed}.{source_group}")
            for metric, value in expected.items():
                if not close(row[metric], value): errors.append(f"组效应错误:{seed}.{source_group}.{metric}")
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        if sha256(output / name) != digest: errors.append(f"产物哈希错误:{name}")
    decision = json.loads((output / "decision.json").read_text())
    if decision.get("classification") != "descriptive_three_group_external_evidence" or not decision.get("valid_result"): errors.append("决策错误")
    report = {"schema": "emotiontalk-group-audit-validator-v1", "status": "failed" if errors else "passed", "run_id": output.name,
              "parameter_snapshot_sha256": sha256(snapshot), "checks": {"hashes": not any("哈希" in error for error in errors),
              "group_effects": not any("组" in error for error in errors), "decision": "决策错误" not in errors}, "errors": errors}
    (output / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(report, ensure_ascii=False)); raise SystemExit(1 if errors else 0)

if __name__ == "__main__": main()
