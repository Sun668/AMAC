#!/usr/bin/env python3
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
def group(clip_id): return str(clip_id).split("/", 1)[0]

def load_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] in ("B4", "H0"):
                rows.append({"seed": int(row["seed"]), "clip_id": row["clip_id"], "group": group(row["clip_id"]),
                             "condition": row["condition"], "errors": int(row["prefinal_errors"]),
                             "commits": int(row["prefinal_commits"]), "revisions": int(row["revisions"]),
                             "covered": int(row["stage2_covered"])})
    return rows

def summarize(rows):
    return {"error": sum(row["errors"] for row in rows) / sum(row["commits"] for row in rows),
            "revision": sum(row["revisions"] for row in rows) / len(rows),
            "coverage": sum(row["covered"] for row in rows) / len(rows)}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    started = time.time(); snapshot = Path(args.parameter_snapshot).resolve(); output = Path(args.condition_dir).resolve()
    if output.exists(): raise RuntimeError("结果目录已存在，禁止覆盖")
    output.mkdir(parents=True); root = Path(__file__).resolve().parents[3]; config = json.loads(snapshot.read_text()); rows = load_rows(root / config["models"]["rows"])
    expected_groups = config["dataset"]["expected_test_groups"]; result_rows = []
    for seed in config["sampling"]["risk_seeds"]:
        for source_group in expected_groups:
            selected = [row for row in rows if row["seed"] == seed and row["group"] == source_group]
            values = {condition: summarize([row for row in selected if row["condition"] == condition]) for condition in ("B4", "H0")}
            result_rows.append({"seed": seed, "group": source_group,
                                "clips": len({row["clip_id"] for row in selected if row["condition"] == "B4"}),
                                "error_reduction": values["B4"]["error"] - values["H0"]["error"],
                                "revision_path_reduction": values["B4"]["revision"] - values["H0"]["revision"],
                                "coverage_gap": abs(values["B4"]["coverage"] - values["H0"]["coverage"])})
    with (output / "group_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    summaries = {}
    for source_group in expected_groups:
        grouped = [row for row in result_rows if row["group"] == source_group]; summaries[source_group] = {"clips": grouped[0]["clips"]}
        for metric in ("error_reduction", "revision_path_reduction", "coverage_gap"):
            values = [row[metric] for row in grouped]; summaries[source_group][metric] = {"min": min(values), "max": max(values), "mean": float(np.mean(values))}
    common = {"run_id": output.name, "parameter_snapshot_sha256": sha256(snapshot), "protocol_version": config["environment"]["protocol_version"]}
    metrics = {"schema": "emotiontalk-group-audit-metrics-v1", "status": "completed", **common, "independent_groups": 3,
               "confidence_interval_reported": False, "reason": "Only three released top-level test groups are available.", "groups": summaries}
    valid = (sorted({row["group"] for row in rows}) == sorted(expected_groups) and sum(value["clips"] for value in summaries.values()) == config["dataset"]["expected_test_clips"] and len(result_rows) == 15)
    decision = {"schema": "emotiontalk-group-audit-decision-v1", "status": "completed", **common, "valid_result": valid,
                "classification": "descriptive_three_group_external_evidence", "claim_boundary": "Consistent with error transfer in three held-out groups; no population-level inferential claim."}
    write(output / "metrics.json", metrics); write(output / "decision.json", decision)
    write(output / "costs.json", {"schema": "emotiontalk-group-audit-costs-v1", "status": "completed", **common, "external_api_usd": 0.0, "wall_seconds": time.time() - started})
    artifacts = {name: sha256(output / name) for name in ("group_effects.csv", "metrics.json", "decision.json", "costs.json")}
    write(output / "manifest.json", {"schema": "emotiontalk-group-audit-manifest-v1", "status": "completed", **common, "source": config["models"], "artifacts": artifacts})
    print(json.dumps({"run_id": output.name, "valid_result": valid, "groups": summaries}, ensure_ascii=False)); raise SystemExit(0 if valid else 1)

if __name__ == "__main__": main()
