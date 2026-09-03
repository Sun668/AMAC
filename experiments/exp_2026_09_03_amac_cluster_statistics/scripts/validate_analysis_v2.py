#!/usr/bin/env python3
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

METRICS = ("error_reduction", "revision_path_reduction", "revision_opportunity_reduction", "coverage_gap")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def close(a, b): return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
def group(clip): return str(clip).split("$_$", 1)[0]

def read_source(path, field):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row[field] in ("B4", "H0"):
                rows.append((int(row["seed"]), row["clip_id"], group(row["clip_id"]), row[field],
                             int(row["prefinal_errors"]), int(row["prefinal_commits"]),
                             int(row["revisions"]), int(row["stage2_covered"])))
    return rows

def arrays_for(rows, groups):
    ordered = sorted(groups)
    arrays = {condition: np.zeros((len(ordered), 5), dtype=float) for condition in ("B4", "H0")}
    lookup = {value: index for index, value in enumerate(ordered)}
    for _, _, source, condition, errors, commits, revisions, covered in rows:
        arrays[condition][lookup[source]] += [errors, commits, revisions, 1, covered]
    return ordered, arrays

def effect(b4, h0):
    return np.asarray([
        b4[..., 0] / b4[..., 1] - h0[..., 0] / h0[..., 1],
        b4[..., 2] / b4[..., 3] - h0[..., 2] / h0[..., 3],
        b4[..., 2] / b4[..., 1] - h0[..., 2] / h0[..., 1],
        np.abs(b4[..., 4] / b4[..., 3] - h0[..., 4] / h0[..., 3]),
    ]).T

def recompute(rows, groups, repetitions, seed):
    ordered, arrays = arrays_for(rows, groups)
    point = effect(arrays["B4"].sum(axis=0), arrays["H0"].sum(axis=0))
    samples = np.random.default_rng(seed).integers(0, len(ordered), size=(repetitions, len(ordered)))
    distribution = effect(arrays["B4"][samples].sum(axis=1), arrays["H0"][samples].sum(axis=1))
    return point, np.quantile(distribution, [0.025, 0.975], axis=0).T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-snapshot", required=True)
    parser.add_argument("--condition-dir", required=True)
    args = parser.parse_args()
    snapshot_path = Path(args.parameter_snapshot).resolve()
    output = Path(args.condition_dir).resolve()
    root = Path(__file__).resolve().parents[3]
    config = json.loads(snapshot_path.read_text())
    errors = []
    required = ("manifest.json", "metrics.json", "decision.json", "cluster_intervals.csv",
                "source_disjoint_group_effects.csv", "costs.json")
    for name in required:
        if not (output / name).is_file(): errors.append(f"缺少产物: {name}")
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    decision = json.loads((output / "decision.json").read_text())
    snapshot_hash = sha256(snapshot_path)
    for kind in ("runner", "validator", "source_audit"):
        if sha256(root / config["environment"][f"{kind}_path"]) != config["environment"][f"{kind}_sha256"]:
            errors.append(f"{kind} 哈希错误")
    for model in ("ridge", "masked_fusion"):
        for kind in ("rows", "validator", "snapshot"):
            if sha256(root / config["models"][model][kind]) != config["models"][model][f"{kind}_sha256"]:
                errors.append(f"{model}.{kind} 哈希错误")
    for document in (manifest, metrics, decision):
        if document.get("run_id") != output.name or document.get("parameter_snapshot_sha256") != snapshot_hash:
            errors.append("产物身份错误")
    for name, digest in manifest.get("artifacts", {}).items():
        if sha256(output / name) != digest: errors.append(f"产物哈希错误: {name}")

    audit = json.loads((root / config["environment"]["source_audit_path"]).read_text())
    disjoint = set(audit["source_disjoint_group_ids"])
    interval_rows = {}
    with (output / "cluster_intervals.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            interval_rows[(row["model"], row["scope"], int(row["risk_seed"]), row["metric"])] = row
    group_rows = {}
    with (output / "source_disjoint_group_effects.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group_rows[(row["model"], int(row["risk_seed"]), row["source_group"])] = row

    for model_index, (model, field) in enumerate((("ridge", "condition"), ("masked_fusion", "policy"))):
        rows = read_source(root / config["models"][model]["rows"], field)
        all_groups = {row[2] for row in rows}
        for scope, groups in (("official_test", all_groups), ("source_disjoint_test", disjoint)):
            for risk_seed in sorted({row[0] for row in rows}):
                selected = [row for row in rows if row[0] == risk_seed and row[2] in groups]
                bootstrap_seed = (config["statistics"]["bootstrap_seed_base"] + model_index * 100000 +
                                  risk_seed + (50000 if scope == "source_disjoint_test" else 0))
                point, intervals = recompute(selected, groups,
                                             config["statistics"]["bootstrap_repetitions"], bootstrap_seed)
                for index, metric in enumerate(METRICS):
                    recorded = interval_rows[(model, scope, risk_seed, metric)]
                    if (not close(point[index], recorded["estimate"]) or
                            not close(intervals[index, 0], recorded["ci95_low"]) or
                            not close(intervals[index, 1], recorded["ci95_high"])):
                        errors.append(f"统计复算错误: {model}.{scope}.{risk_seed}.{metric}")
                disjoint_rows = [row for row in rows if row[0] == risk_seed and row[2] in disjoint]
                ordered, grouped = arrays_for(disjoint_rows, disjoint)
                individual = effect(grouped["B4"], grouped["H0"])
                for index, source in enumerate(ordered):
                    recorded = group_rows[(model, risk_seed, source)]
                    if int(recorded["clips"]) != int(grouped["B4"][index, 3]):
                        errors.append(f"组样本数错误: {model}.{risk_seed}.{source}")
                    for metric_index, metric in enumerate(METRICS):
                        if not close(individual[index, metric_index], recorded[metric]):
                            errors.append(f"组效应复算错误: {model}.{risk_seed}.{source}.{metric}")

    validity = metrics.get("validity_checks", {})
    if (decision.get("valid_result") != all(validity.values()) or
            decision.get("classification") != "posthoc_statistical_correction" or
            decision.get("supersedes_clip_bootstrap_for_cross_source_uncertainty") is not True):
        errors.append("决策错误")
    report = {
        "schema": "amac-cluster-statistics-validator-v2", "status": "failed" if errors else "passed",
        "run_id": output.name, "parameter_snapshot_sha256": snapshot_hash,
        "protocol_version": config["environment"]["protocol_version"],
        "checks": {
            "hashes": not any("哈希" in error for error in errors),
            "identity": not any("身份" in error for error in errors),
            "cluster_intervals": not any("统计复算" in error for error in errors),
            "per_group_effects": not any("组效应复算" in error or "组样本数" in error for error in errors),
            "decision": not any("决策" in error for error in errors),
        },
        "errors": errors,
    }
    (output / "validator.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()
