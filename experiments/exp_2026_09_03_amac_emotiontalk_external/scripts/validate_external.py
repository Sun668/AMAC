#!/usr/bin/env python3
import argparse,csv,hashlib,importlib.util,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while c:=f.read(8*1024*1024):h.update(c)
    return h.hexdigest()
def close(a,b): return a is b if a is None or b is None else math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-7)
def agg(rows):
    c=sum(x["prefinal_commits"] for x in rows);e=sum(x["prefinal_errors"] for x in rows);return {"paths":len(rows),"prefinal_commits":c,"prefinal_errors":e,"prefinal_committed_error_rate":e/c if c else None,"committed_revision_rate":sum(x["revisions"] for x in rows)/len(rows),"premature_exposure_rate":sum(x["premature"] for x in rows)/len(rows),"stage_two_coverage":sum(x["stage2_covered"] for x in rows)/len(rows),"time_to_first_commit":sum(x["time_to_first"] for x in rows)/len(rows),"final_state_identity":sum(x["final_identity"] for x in rows)/len(rows)}
def main():
    p=argparse.ArgumentParser();p.add_argument("--parameter-snapshot",required=True);p.add_argument("--condition-dir",required=True);a=p.parse_args();sp=Path(a.parameter_snapshot).resolve();out=Path(a.condition_dir).resolve();root=Path(__file__).resolve().parents[3];s=json.loads(sp.read_text());errors=[];runner=load("runner",root/s["environment"]["runner_path"])
    for key in ("runner","validator","index","index_manifest","development_runner"):
        if sha(root/s["environment"][f"{key}_path"])!=s["environment"][f"{key}_sha256"]:errors.append(f"{key} 哈希错误")
    required=["manifest.json","metrics.json","decision.json","costs.json","per_path.csv","predictions.npz","models.pt"]
    for x in required:
        if not (out/x).is_file():errors.append(f"缺少产物: {x}")
    manifest=json.loads((out/"manifest.json").read_text());metrics=json.loads((out/"metrics.json").read_text());decision=json.loads((out/"decision.json").read_text());sh=sha(sp)
    for name,d in (("manifest",manifest),("metrics",metrics),("decision",decision)):
        if d.get("parameter_snapshot_sha256")!=sh or d.get("run_id")!=out.name:errors.append(f"{name} 身份错误")
    if set(manifest.get("used_groups",{}).get("train",[]))&set(manifest.get("used_groups",{}).get("test",[])):errors.append("训练测试组重叠")
    for name,digest in manifest.get("artifacts",{}).items():
        if sha(out/name)!=digest:errors.append(f"产物哈希错误: {name}")
    rows=[]
    with (out/"per_path.csv").open(newline="") as f:
        for x in csv.DictReader(f):
            for k in ("seed","gold_state","final_state","prefinal_commits","prefinal_errors","revisions","premature","time_to_first","stage2_covered","final_identity"):x[k]=int(x[k])
            rows.append(x)
    seeds=s["sampling"]["seeds"];conditions=s["policy"]["conditions"];expected=len(seeds)*s["dataset"]["expected_test"]*len(s["sampling"]["arrival_orders"])*len(conditions);ids=Counter((x["seed"],x["clip_id"],x["path"],x["condition"]) for x in rows)
    if len(rows)!=expected or len(ids)!=expected or any(v!=1 for v in ids.values()):errors.append("轨迹身份错误")
    recomputed={}
    for seed in seeds:
        recomputed[str(seed)]={}
        for condition in conditions:
            value=agg([x for x in rows if x["seed"]==seed and x["condition"]==condition]);recomputed[str(seed)][condition]=value
            for field,v in value.items():
                if not close(v,metrics["per_seed"][str(seed)]["conditions"][condition][field]):errors.append(f"指标错误:{seed}.{condition}.{field}")
    for seed in seeds:
        sr=[x for x in rows if x["seed"]==seed]
        for comparator in ("B2","B3","B4"):
            own,other=recomputed[str(seed)]["H0"],recomputed[str(seed)][comparator];e={"error_reduction":other["prefinal_committed_error_rate"]-own["prefinal_committed_error_rate"],"revision_relative_reduction":(other["committed_revision_rate"]-own["committed_revision_rate"])/other["committed_revision_rate"] if other["committed_revision_rate"] else None};e.update(runner.intervals(sr,comparator,s["statistics"]["bootstrap_repetitions"],seed*100+len(comparator)))
            for field,v in e.items():
                r=metrics["per_seed"][str(seed)]["comparisons"][comparator][field]
                if isinstance(v,list):
                    if any(not close(x,y) for x,y in zip(v,r)):errors.append(f"比较区间错误:{seed}.{comparator}")
                elif not close(v,r):errors.append(f"比较错误:{seed}.{comparator}.{field}")
    with np.load(out/"predictions.npz",allow_pickle=False) as z:
        gold,final=z["gold"],z["final"];anchor={"accuracy":float(np.mean(gold==final)),"macro_f1":float(f1_score(gold,final,average="macro")),"weighted_f1":float(f1_score(gold,final,average="weighted"))}
        for k,v in anchor.items():
            if not close(v,metrics["anchor"][k]):errors.append(f"锚点错误:{k}")
    g=s["gates"];vals=lambda c,f:[metrics["per_seed"][str(seed)]["comparisons"][c][f] for seed in seeds];checks={"coverage":all(recomputed[str(seed)]["H0"]["stage_two_coverage"]>=g["coverage_min"] for seed in seeds),"error_direction":sum(v>0 for v in vals("B4","error_reduction"))>=g["direction_required_seeds"] and np.mean(vals("B4","error_reduction"))>0,"revision_direction":sum(v>0 for v in vals("B4","revision_relative_reduction"))>=g["direction_required_seeds"] and np.mean(vals("B4","revision_relative_reduction"))>0,"b2_direction":sum(v>0 for v in vals("B2","error_reduction"))>=g["direction_required_seeds"],"final_identity":all(recomputed[str(seed)][c]["final_state_identity"]==1 for seed in seeds for c in conditions)}
    if checks!=metrics.get("checks") or decision.get("direction_consistent")!=all(checks.values()):errors.append("决策重算错误")
    report={"schema":"amac-emotiontalk-validator-v1","status":"failed" if errors else "passed","run_id":out.name,"parameter_snapshot_sha256":sh,"protocol_version":s["environment"]["protocol_version"],"errors":errors};(out/"validator.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps(report,ensure_ascii=False));raise SystemExit(1 if errors else 0)
if __name__=="__main__":main()
