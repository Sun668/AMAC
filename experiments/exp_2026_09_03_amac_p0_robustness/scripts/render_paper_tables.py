#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ALL = ("B0", "B1", "B2", "B3", "B4", "H0", "O1", "LR", "PLATT", "ISO")
LABEL = {"PLATT": "Platt", "ISO": "Isotonic"}
EOL = chr(92) * 2


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}"


def mean(metrics, scope, condition, field):
    return metrics[scope][condition]["summary"][field]["mean"]


def pm(metrics, scope, condition, field, scale=100, digits=2):
    value = metrics[scope][condition]["summary"][field]
    if value["mean"] is None:
        return "N/A"
    if value["std"] is None:
        return f"{scale * value['mean']:.{digits}f}"
    return (f"\\({scale * value['mean']:.{digits}f}"
            f"\\pm{scale * value['std']:.{digits}f}\\)")


def save(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-dir", required=True)
    args = parser.parse_args()
    output = Path(args.condition_dir).resolve()
    validator = json.loads((output / "validator.json").read_text(encoding="utf-8"))
    if not validator.get("accepted"):
        raise RuntimeError("结果未通过独立验证，禁止生成论文表格")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    split = metrics["source_split_audit"]
    paper = Path(__file__).resolve().parents[3] / "paper" / "generated"
    paper.mkdir(parents=True, exist_ok=True)
    save(paper / "p0_values.tex", [
        f"\\newcommand{{\\SourceDisjointGroups}}{{{split['source_disjoint_test_groups']}}}",
        f"\\newcommand{{\\SourceDisjointClips}}{{{split['source_disjoint_test_clips']}}}",
        f"\\newcommand{{\\SourceDisjointFraction}}{{{100 * split['source_disjoint_test_fraction']:.2f}\\%}}",
        f"\\newcommand{{\\OverlappingSourceGroups}}{{{split['overlapping_source_groups']}}}",
    ])

    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Complete CH-SIMS v2 operating-point report. H0 is mean \\(\\pm\\) sample standard deviation over five seeds; deterministic policies are reported once. Error is undefined when no preterminal state is committed. Differences in prose are computed from unrounded values.}",
        "\\label{tab:all-policies}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Policy & Error (\\%) & State cov. (\\%) & Stage-2 cov. (\\%) & Revision/path (\\%) & Revision/opportunity (\\%) & \\(T_f\\) " + EOL,
        "\\midrule",
    ]
    for condition in ("B0", "B1", "B2", "B3", "B4", "H0", "O1"):
        values = [
            pm(metrics, "official_test", condition, "prefinal_committed_error_rate"),
            pm(metrics, "official_test", condition, "committed_state_coverage"),
            pm(metrics, "official_test", condition, "stage_two_path_coverage"),
            pm(metrics, "official_test", condition, "committed_revision_rate_per_path"),
            pm(metrics, "official_test", condition, "revision_rate_per_opportunity"),
            pm(metrics, "official_test", condition, "time_to_first_commit", 1, 3),
        ]
        lines.append(f"{condition} & " + " & ".join(values) + " " + EOL)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    save(paper / "all_policies_table.tex", lines)

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Paired clip-level bootstrap intervals for H0 minus B4 on the official test. Positive values favor H0. Each seed uses 5,000 resamples; all six paths remain grouped with the clip.}",
        "\\label{tab:primary-ci}", "\\begin{tabular}{crr}", "\\toprule",
        "Seed & Error reduction (pp) & Revision reduction (pp) " + EOL,
        "\\midrule",
    ]
    for seed, values in metrics["paired_bootstrap_h0_vs_b4"]["official_test"].items():
        error = values["error_absolute_reduction"]
        revision = values["revision_per_path_absolute_reduction"]
        lines.append(
            f"{seed} & {100*error['estimate']:.2f} "
            f"[{100*error['ci95'][0]:.2f}, {100*error['ci95'][1]:.2f}] & "
            f"{100*revision['estimate']:.2f} "
            f"[{100*revision['ci95'][0]:.2f}, {100*revision['ci95'][1]:.2f}] {EOL}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save(paper / "primary_ci_table.tex", lines)

    grouped = {}
    for key, value in metrics["risk_coverage"]["curves"].items():
        grouped.setdefault(key.split(":")[0], []).append(value)
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Descriptive risk--coverage summary. AURC is normalized over 75--100\\% committed-state coverage; lower is better. Thresholds are swept after the sealed test and are not deployment settings.}",
        "\\label{tab:aurc}", "\\begin{tabular}{lrrr}", "\\toprule",
        "Estimator & AURC & Error@80\\% & Error@90\\% " + EOL,
        "\\midrule",
    ]
    for condition in ("B2", "B4", "H0", "LR", "PLATT", "ISO"):
        values = grouped[condition]
        available = lambda field: [item[field] for item in values if item[field] is not None]
        average = lambda field: (sum(available(field)) / len(available(field))
                                 if available(field) else None)
        lines.append(
            f"{LABEL.get(condition, condition)} & {pct(average('normalized_aurc'))} & "
            f"{pct(average('error_at_0_80'))} & {pct(average('error_at_0_90'))} {EOL}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save(paper / "aurc_table.tex", lines)

    lines = [
        "\\begin{table*}[t]", "\\centering",
        f"\\caption{{Source-video-disjoint robustness subset: {split['source_disjoint_test_clips']} clips from {split['source_disjoint_test_groups']} test-only source groups. H0 is mean \\(\\pm\\) sample standard deviation over five seeds. This post-hoc subset is supplementary.}}",
        "\\label{tab:source-disjoint}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Policy & Error (\\%) & State cov. (\\%) & Stage-2 cov. (\\%) & Revision/path (\\%) & Revision/opportunity (\\%) & \\(T_f\\) " + EOL,
        "\\midrule",
    ]
    for condition in ("B4", "H0", "LR", "PLATT", "ISO"):
        values = [
            pm(metrics, "source_disjoint_test", condition, "prefinal_committed_error_rate"),
            pm(metrics, "source_disjoint_test", condition, "committed_state_coverage"),
            pm(metrics, "source_disjoint_test", condition, "stage_two_path_coverage"),
            pm(metrics, "source_disjoint_test", condition, "committed_revision_rate_per_path"),
            pm(metrics, "source_disjoint_test", condition, "revision_rate_per_opportunity"),
            pm(metrics, "source_disjoint_test", condition, "time_to_first_commit", 1, 3),
        ]
        lines.append(f"{LABEL.get(condition, condition)} & " + " & ".join(values) + " " + EOL)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    save(paper / "source_disjoint_table.tex", lines)

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Paired source-disjoint effects for H0 minus B4. Positive values favor H0. Intervals use 5,000 clip-level resamples.}",
        "\\label{tab:source-disjoint-ci}", "\\begin{tabular}{crr}", "\\toprule",
        "Seed & Error reduction (pp) & Revision/opportunity (pp) " + EOL,
        "\\midrule",
    ]
    for seed, values in metrics["paired_bootstrap_h0_vs_b4"]["source_disjoint_test"].items():
        error = values["error_absolute_reduction"]
        revision = values["revision_per_opportunity_absolute_reduction"]
        lines.append(
            f"{seed} & {100*error['estimate']:.2f} "
            f"[{100*error['ci95'][0]:.2f}, {100*error['ci95'][1]:.2f}] & "
            f"{100*revision['estimate']:.2f} "
            f"[{100*revision['ci95'][0]:.2f}, {100*revision['ci95'][1]:.2f}] {EOL}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save(paper / "source_disjoint_ci_table.tex", lines)

    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Stage-specific CH-SIMS v2 denominators at selected operating points. Coverage and error are computed separately within each stage. H0 is averaged across five seeds.}",
        "\\label{tab:stage-specific}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Policy & Stage-1 count & Stage-1 cov. (\\%) & Stage-1 error (\\%) & Stage-2 count & Stage-2 cov. (\\%) & Stage-2 error (\\%) " + EOL,
        "\\midrule",
    ]
    for condition in ALL:
        values = [
            f"{mean(metrics, 'official_test', condition, 'stage1_committed_count'):.0f}",
            pm(metrics, "official_test", condition, "stage1_coverage"),
            pm(metrics, "official_test", condition, "stage1_committed_error_rate"),
            f"{mean(metrics, 'official_test', condition, 'stage2_committed_count'):.0f}",
            pm(metrics, "official_test", condition, "stage2_coverage"),
            pm(metrics, "official_test", condition, "stage2_committed_error_rate"),
        ]
        lines.append(f"{LABEL.get(condition, condition)} & " + " & ".join(values) + " " + EOL)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    save(paper / "stage_specific_table.tex", lines)
    print(json.dumps({"generated_dir": str(paper), "files": 6}, ensure_ascii=False))


if __name__ == "__main__":
    main()
