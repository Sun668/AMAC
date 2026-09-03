#!/usr/bin/env python3
import argparse, csv, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t

METRICS = ("error_reduction", "revision_path_reduction", "revision_opportunity_reduction", "coverage_gap")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def source_group(clip_id): return str(clip_id).split("$_$", 1)[0]

def load_rows(path, condition_field):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            condition = raw[condition_field]
            if condition not in ("B4", "H0"): continue
            rows.append({"seed": int(raw["seed"]), "clip_id": raw["clip_id"], "group": source_group(raw["clip_id"]), "path": raw["path"], "condition": condition, "commits": int(raw["prefinal_commits"]), "errors": int(raw["prefinal_errors"]), "revisions": int(raw["revisions"]), "covered": int(raw["stage2_covered"])})
    return rows

def aggregate_groups(rows, groups):
    grouped = {condition: {group: np.zeros(5, dtype=np.float64) for group in groups} for condition in ("B4", "H0")}
    for row in rows:
        if row["group"] not in groups: continue
        grouped[row["condition"]][row["group"]] += np.asarray([row["errors"], row["commits"], row["revisions"], 1, row["covered"]], dtype=np.float64)
    return grouped

def effects_from_sums(b4, h0):
    return np.asarray([b4[0] / b4[1] - h0[0] / h0[1], b4[2] / b4[3] - h0[2] / h0[3], b4[2] / b4[1] - h0[2] / h0[1], abs(b4[4] / b4[3] - h0[4] / h0[3])], dtype=np.float64)

def cluster_analysis(rows, groups, repetitions, seed):
    ordered = sorted(groups); grouped = aggregate_groups(rows, set(ordered)); b4 = np.stack([grouped["B4"][group] for group in ordered]); h0 = np.stack([grouped["H0"][group] for group in ordered])
    if np.any(b4[:, 3] == 0) or np.any(h0[:, 3] == 0): raise RuntimeError("存在缺失条件的 source group")
    point = effects_from_sums(b4.sum(axis=0), h0.sum(axis=0)); rng = np.random.default_rng(seed); samples = rng.integers(0, len(ordered), size=(repetitions, len(ordered)))
    b4_sum = b4[samples].sum(axis=1); h0_sum = h0[samples].sum(axis=1)
    distributions = np.column_stack((b4_sum[:, 0] / b4_sum[:, 1] - h0_sum[:, 0] / h0_sum[:, 1], b4_sum[:, 2] / b4_sum[:, 3] - h0_sum[:, 2] / h0_sum[:, 3], b4_sum[:, 2] / b4_sum[:, 1] - h0_sum[:, 2] / h0_sum[:, 1], np.abs(b4_sum[:, 4] / b4_sum[:, 3] - h0_sum[:, 4] / h0_sum[:, 3])))
    intervals = np.quantile(distributions, [0.025, 0.975], axis=0).T
    per_group = {group: effects_from_sums(b4[index], h0[index]) for index, group in enumerate(ordered)}
    return point, intervals, per_group

def exact_sign_test(values):
    values = np.asarray(values); positives = int(np.sum(values > 0)); negatives = int(np.sum(values < 0)); n = positives + negatives
    if n == 0: return {"positive": positives, "negative": negatives, "ties": len(values), "p_two_sided": 1.0}
    tail = min(positives, negatives); p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n))
    return {"positive": positives, "negative": negatives, "ties": int(len(values) - n), "p_two_sided": p}

def group_summary(values):
    values = np.asarray(values, dtype=float); mean = float(values.mean()); sem = float(values.std(ddof=1) / math.sqrt(len(values))); critical = float(t.ppf(0.975, len(values) - 1))
    return {"mean": mean, "median": float(np.median(values)), "q1": float(np.quantile(values, 0.25)), "q3": float(np.quantile(values, 0.75)), "min": float(values.min()), "max": float(values.max()), "mean_t_ci95": [mean - critical * sem, mean + critical * sem], "sign_test": exact_sign_test(values)}

def write_json(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--parameter-snapshot", required=True); parser.add_argument("--condition-dir", required=True); args = parser.parse_args()
    started = time.time(); snapshot_path = Path(args.parameter_snapshot).resolve(); output = Path(args.condition_dir).resolve(); root = Path(__file__).resolve().parents[3]
    if output.exists(): raise RuntimeError("输出目录已存在，禁止覆盖")
    config = json.loads(snapshot_path.read_text());
    if config.get("status") != "frozen": raise RuntimeError("参数快照未冻结")
    for kind in ("runner", "validator", "source_audit"):
        if sha256(root / config["environment"][f"{kind}_path"]) != config["environment"][f"{kind}_sha256"]: raise RuntimeError(f"{kind} 哈希不一致")
    for model in ("ridge", "masked_fusion"):
        for kind in ("rows", "validator", "snapshot"):
            if sha256(root / config["models"][model][kind]) != config["models"][model][f"{kind}_sha256"]: raise RuntimeError(f"{model}.{kind} 哈希不一致")
    audit = json.loads((root / config["environment"]["source_audit_path"]).read_text()); disjoint_groups = set(audit["source_disjoint_group_ids"]); output.mkdir(parents=True)
    results, interval_rows, group_rows = {}, [], []
    for model_index, (model, field) in enumerate((("ridge", "condition"), ("masked_fusion", "policy"))):
        rows = load_rows(root / config["models"][model]["rows"], field); clips = sorted({row["clip_id"] for row in rows}); groups = sorted({row["group"] for row in rows}); seeds = sorted({row["seed"] for row in rows}); results[model] = {}
        for scope, selected_groups in (("official_test", set(groups)), ("source_disjoint_test", disjoint_groups)):
            selected_clips = {row["clip_id"] for row in rows if row["group"] in selected_groups}; scope_result = {"groups": len(selected_groups), "clips": len(selected_clips), "per_seed": {}}
            for risk_seed in seeds:
                selected_rows = [row for row in rows if row["seed"] == risk_seed and row["group"] in selected_groups]; bootstrap_seed = config["statistics"]["bootstrap_seed_base"] + model_index * 100000 + risk_seed + (50000 if scope == "source_disjoint_test" else 0)
                point, intervals, per_group = cluster_analysis(selected_rows, selected_groups, config["statistics"]["bootstrap_repetitions"], bootstrap_seed); seed_result = {}
                for metric_index, metric in enumerate(METRICS):
                    seed_result[metric] = {"estimate": float(point[metric_index]), "cluster_bootstrap_ci95": [float(intervals[metric_index, 0]), float(intervals[metric_index, 1])]}; interval_rows.append({"model": model, "scope": scope, "risk_seed": risk_seed, "source_groups": len(selected_groups), "clips": len(selected_clips), "metric": metric, "estimate": point[metric_index], "ci95_low": intervals[metric_index, 0], "ci95_high": intervals[metric_index, 1], "bootstrap_seed": bootstrap_seed})
                if scope == "source_disjoint_test":
                    seed_result["equal_group_diagnostics"] = {}
                    for metric_index, metric in enumerate(METRICS): seed_result["equal_group_diagnostics"][metric] = group_summary([value[metric_index] for value in per_group.values()])
                    clip_counts = {group: len({row["clip_id"] for row in selected_rows if row["group"] == group}) for group in selected_groups}
                    for group, values in per_group.items(): group_rows.append({"model": model, "risk_seed": risk_seed, "source_group": group, "clips": clip_counts[group], **{metric: values[index] for index, metric in enumerate(METRICS)}})
                scope_result["per_seed"][str(risk_seed)] = seed_result
            results[model][scope] = scope_result
    validity = {"ridge_official_groups": results["ridge"]["official_test"]["groups"] == config["dataset"]["expected_test_groups"], "ridge_source_disjoint_identity": results["ridge"]["source_disjoint_test"]["groups"] == config["dataset"]["expected_source_disjoint_groups"] and results["ridge"]["source_disjoint_test"]["clips"] == config["dataset"]["expected_source_disjoint_clips"], "masked_source_disjoint_identity": results["masked_fusion"]["source_disjoint_test"]["groups"] == config["dataset"]["expected_source_disjoint_groups"] and results["masked_fusion"]["source_disjoint_test"]["clips"] == config["dataset"]["expected_source_disjoint_clips"]}
    common = {"run_id": output.name, "parameter_snapshot_sha256": sha256(snapshot_path), "protocol_version": config["environment"]["protocol_version"]}; metrics = {"schema": "amac-cluster-statistics-metrics-v1", "status": "completed", **common, "direction": "B4_minus_H0_positive_favors_H0", "results": results, "validity_checks": validity, "interpretation": {"official_test": "official-split clustered uncertainty, not source-disjoint generalization", "source_disjoint_test": "limited 15-group post-hoc robustness"}}
    write_json(output / "metrics.json", metrics); write_json(output / "decision.json", {"schema": "amac-cluster-statistics-decision-v1", "status": "completed", **common, "valid_result": all(validity.values()), "classification": "posthoc_statistical_correction", "supersedes_clip_bootstrap_for_cross_source_uncertainty": True})
    with (output / "cluster_intervals.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0])); writer.writeheader(); writer.writerows(interval_rows)
    with (output / "source_disjoint_group_effects.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(group_rows[0])); writer.writeheader(); writer.writerows(group_rows)
    write_json(output / "costs.json", {"schema": "amac-cluster-statistics-costs-v1", "status": "completed", **common, "external_api_usd": 0.0, "wall_seconds": time.time() - started})
    artifacts = ["metrics.json", "decision.json", "cluster_intervals.csv", "source_disjoint_group_effects.csv", "costs.json"]; write_json(output / "manifest.json", {"schema": "amac-cluster-statistics-manifest-v1", "status": "completed_unvalidated", **common, "artifacts": {name: sha256(output / name) for name in artifacts}})
    print(json.dumps({"run_id": output.name, "valid_result": all(validity.values()), "validity_checks": validity}, ensure_ascii=False))

if __name__ == "__main__": main()
