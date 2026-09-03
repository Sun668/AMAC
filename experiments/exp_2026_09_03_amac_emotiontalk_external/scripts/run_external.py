#!/usr/bin/env python3
import argparse, copy, csv, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

ORDERS=("TAV","TVA","ATV","AVT","VTA","VAT")

def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while c:=f.read(8*1024*1024): h.update(c)
    return h.hexdigest()
def fuse(row,visible,classes):
    votes=np.zeros(len(classes)); certainty=[]; labels=[]
    for modality in visible:
        sensor=row[modality]; weight=float(sensor["agreement"])*float(sensor["confidence"]); votes[classes.index(sensor["label"])]+=weight; certainty.append(weight); labels.append(sensor["label"])
    order=np.argsort(votes); top=int(order[-1]); second=float(votes[order[-2]])
    confidence=float(votes[top]/len(visible)); margin=float((votes[top]-second)/len(visible)); consensus=max(labels.count(value) for value in set(labels))/len(labels)
    features=np.asarray([confidence,margin,float(np.mean(certainty)),consensus,len(visible)/3.0]+[float(m in visible) for m in "TAV"]+[float(i==top) for i in range(len(classes))],dtype=np.float32)
    return top,confidence,features
def build_events(rows,orders,classes):
    features=[]; targets=[]; metadata=[]; paths=[]
    for index,row in enumerate(rows):
        gold=classes.index(row["gold"])
        for oi,order in enumerate(orders):
            states=[]; raw=[]
            for stage in (0,1,2):
                state,confidence,vector=fuse(row,order[:stage+1],classes); states.append(state); raw.append(confidence)
                if stage<2:
                    features.append(vector); value=float(state==gold); targets.append([value,value]); metadata.append((index,oi,stage,row["clip_id"],row["group"]))
            paths.append((index,row["clip_id"],order,gold,states,raw))
    return np.asarray(features),np.asarray(targets,dtype=np.float32),metadata,paths
def score_map(metadata,values): return {(x[0],x[1],x[2]):float(values[i]) for i,x in enumerate(metadata)}
def make_records(paths,scores):
    return [(clip,order,gold,states,raw,[scores[(index,ORDERS.index(order),s)] for s in (0,1)]+[1.0]) for index,clip,order,gold,states,raw in paths]
def commits(policy,states,raw,scores,gold,threshold=.5,margin=0):
    if policy=="B0": return list(states)
    result=[None,None,states[2]]
    if policy=="B1": return result
    if policy=="B3":
        if states[0]==states[1]: result[1]=states[1]
        return result
    if policy=="O1":
        held=None
        for s in (0,1):
            if states[s]==gold: held=states[s]
            result[s]=held
        return result
    values=raw if policy in ("B2","B4") else scores; held=None
    for s in (0,1):
        if held is None and values[s]>=threshold: held=states[s]
        elif held is not None and states[s]!=held and values[s]>=threshold+(margin if policy in ("B4","H0") else 0): held=states[s]
        result[s]=held
    return result
def evaluate(records,condition,params):
    out=[]
    for clip,order,gold,states,raw,scores in records:
        c=commits(condition,states,raw,scores,gold,params.get("threshold",.5),params.get("margin",0)); pre=[x for x in c[:2] if x is not None]; rev=0; previous=None
        for value in c:
            if value is not None and previous is not None and value!=previous: rev+=1
            if value is not None: previous=value
        out.append({"clip_id":clip,"path":order,"condition":condition,"gold_state":gold,"final_state":states[2],"commits":json.dumps(c,separators=(",",":")),"prefinal_commits":len(pre),"prefinal_errors":sum(x!=gold for x in pre),"revisions":rev,"premature":int(any(x!=gold for x in pre) and states[2]==gold),"time_to_first":next((i+1 for i,x in enumerate(c) if x is not None),3),"stage2_covered":int(any(x is not None for x in c[:2])),"final_identity":int(c[2]==states[2])})
    return out
def aggregate(rows):
    c=sum(x["prefinal_commits"] for x in rows); e=sum(x["prefinal_errors"] for x in rows)
    return {"paths":len(rows),"prefinal_commits":c,"prefinal_errors":e,"prefinal_committed_error_rate":e/c if c else None,"committed_revision_rate":sum(x["revisions"] for x in rows)/len(rows),"premature_exposure_rate":sum(x["premature"] for x in rows)/len(rows),"stage_two_coverage":sum(x["stage2_covered"] for x in rows)/len(rows),"time_to_first_commit":sum(x["time_to_first"] for x in rows)/len(rows),"final_state_identity":sum(x["final_identity"] for x in rows)/len(rows)}
def tune(records,condition,policy):
    candidates=[]
    for threshold in policy["threshold_grid"]:
        for margin in ([0.0] if condition=="B2" else policy["margin_grid"]):
            m=aggregate(evaluate(records,condition,{"threshold":threshold,"margin":margin}))
            if m["stage_two_coverage"]<policy["coverage_target"] or m["prefinal_committed_error_rate"] is None: continue
            objective=m["prefinal_committed_error_rate"]+policy["revision_weight"]*m["committed_revision_rate"]+policy["wait_weight"]*(m["time_to_first_commit"]-1); candidates.append((objective,abs(m["stage_two_coverage"]-policy["coverage_target"]),-threshold,margin))
    if not candidates: raise RuntimeError(f"{condition} 没有满足覆盖率的参数")
    x=min(candidates); return {"threshold":float(-x[2]),"margin":float(x[3]),"objective":float(x[0])}
def intervals(rows,comparator,repetitions,seed):
    clips=sorted({x["clip_id"] for x in rows}); pos={x:i for i,x in enumerate(clips)}; arrays={}
    for condition in ("H0",comparator):
        a=np.zeros((len(clips),4))
        for x in rows:
            if x["condition"]==condition: a[pos[x["clip_id"]]] += [x["prefinal_commits"],x["prefinal_errors"],x["revisions"],1]
        arrays[condition]=a
    rng=np.random.default_rng(seed); er=[]; rr=[]
    for _ in range(repetitions):
        sample=rng.integers(0,len(clips),len(clips)); m=arrays["H0"][sample].sum(0); c=arrays[comparator][sample].sum(0); er.append(c[1]/c[0]-m[1]/m[0]); cr=c[2]/c[3]; rr.append((cr-m[2]/m[3])/cr if cr else np.nan)
    return {"error_reduction_ci95":[float(np.nanquantile(er,.025)),float(np.nanquantile(er,.975))],"revision_relative_reduction_ci95":[float(np.nanquantile(rr,.025)),float(np.nanquantile(rr,.975))]}
def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--parameter-snapshot",required=True); p.add_argument("--condition-dir",required=True); a=p.parse_args(); started=time.time(); sp=Path(a.parameter_snapshot).resolve(); out=Path(a.condition_dir).resolve()
    if out.exists(): raise RuntimeError("输出目录已存在，禁止覆盖")
    s=json.loads(sp.read_text()); root=Path(__file__).resolve().parents[3]
    for key in ("runner","validator","index","index_manifest","development_runner"):
        if sha(root/s["environment"][f"{key}_path"])!=s["environment"][f"{key}_sha256"]: raise RuntimeError(f"{key} 哈希不一致")
    rows=[json.loads(line) for line in (root/s["environment"]["index_path"]).open()]; train=[x for x in rows if x["group"] in s["dataset"]["train_groups"]]; test=[x for x in rows if x["group"] in s["dataset"]["test_groups"]]
    if len(train)!=s["dataset"]["expected_train"] or len(test)!=s["dataset"]["expected_test"]: raise RuntimeError("分组样本数不一致")
    out.mkdir(parents=True); base=load_module("dev",root/s["environment"]["development_runner_path"]); classes=s["dataset"]["classes"]; orders=s["sampling"]["arrival_orders"]
    train_x,train_y,train_meta,train_paths=build_events(train,orders,classes); test_x,_,test_meta,test_paths=build_events(test,orders,classes); cfg=dict(s["models"]["risk_predictor"]); cfg.update(s["training"]); policy=s["policy"]
    all_rows=[]; per_seed={}; models={}; risks=[]
    for seed in s["sampling"]["seeds"]:
        np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True); model,scaler,_,cal_mask,training=base.train_amac(train_x,train_y,train_meta,cfg,seed); _,train_heads=base.predict_scores(model,scaler,train_x); _,test_heads=base.predict_scores(model,scaler,test_x)
        cal_indices={x[0] for i,x in enumerate(train_meta) if cal_mask[i]}; cal_clip_ids={train[i]["clip_id"] for i in cal_indices}; train_scores=score_map(train_meta,train_heads[:,0]); test_scores=score_map(test_meta,test_heads[:,0]); cal_records=[r for r in make_records(train_paths,train_scores) if r[0] in cal_clip_ids]; test_records=make_records(test_paths,test_scores)
        params={x:{"threshold":.5,"margin":0} for x in policy["conditions"]}; params["B2"]=tune(cal_records,"B2",policy); params["B4"]=tune(cal_records,"B4",policy); params["H0"]=tune(cal_records,"H0",policy)
        seed_rows=[]
        for condition in policy["conditions"]: seed_rows.extend(evaluate(test_records,condition,params[condition]))
        for x in seed_rows: x["seed"]=seed
        all_rows.extend(seed_rows); cm={condition:aggregate([x for x in seed_rows if x["condition"]==condition]) for condition in policy["conditions"]}; comparisons={}
        for comparator in ("B2","B3","B4"):
            own,other=cm["H0"],cm[comparator]; effect={"error_reduction":other["prefinal_committed_error_rate"]-own["prefinal_committed_error_rate"],"revision_relative_reduction":(other["committed_revision_rate"]-own["committed_revision_rate"])/other["committed_revision_rate"] if other["committed_revision_rate"] else None}; effect.update(intervals(seed_rows,comparator,s["statistics"]["bootstrap_repetitions"],seed*100+len(comparator))); comparisons[comparator]=effect
        per_seed[str(seed)]={"conditions":cm,"comparisons":comparisons,"parameters":params,"training":training}; models[str(seed)]={"state_dict":copy.deepcopy(model.state_dict()),"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_}; risks.append(test_heads[:,0])
    fields=["prefinal_committed_error_rate","committed_revision_rate","premature_exposure_rate","stage_two_coverage","time_to_first_commit"]; aggregate_summary={}
    for condition in policy["conditions"]:
        aggregate_summary[condition]={}
        for field in fields:
            vals=[per_seed[str(seed)]["conditions"][condition][field] for seed in s["sampling"]["seeds"]]; aggregate_summary[condition][field]={"mean":None,"std":None} if any(v is None for v in vals) else {"mean":float(np.mean(vals)),"std":float(np.std(vals,ddof=1))}
    gold=np.asarray([classes.index(x["gold"]) for x in test]); final=np.asarray([fuse(x,"TAV",classes)[0] for x in test]); anchor={"accuracy":float(np.mean(gold==final)),"macro_f1":float(f1_score(gold,final,average="macro")),"weighted_f1":float(f1_score(gold,final,average="weighted"))}
    seeds=s["sampling"]["seeds"]; g=s["gates"]; vals=lambda c,f:[per_seed[str(seed)]["comparisons"][c][f] for seed in seeds]; checks={"coverage":bool(all(per_seed[str(seed)]["conditions"]["H0"]["stage_two_coverage"]>=g["coverage_min"] for seed in seeds)),"error_direction":bool(sum(v>0 for v in vals("B4","error_reduction"))>=g["direction_required_seeds"] and np.mean(vals("B4","error_reduction"))>0),"revision_direction":bool(sum(v>0 for v in vals("B4","revision_relative_reduction"))>=g["direction_required_seeds"] and np.mean(vals("B4","revision_relative_reduction"))>0),"b2_direction":bool(sum(v>0 for v in vals("B2","error_reduction"))>=g["direction_required_seeds"]),"final_identity":bool(all(per_seed[str(seed)]["conditions"][c]["final_state_identity"]==1 for seed in seeds for c in policy["conditions"]))}
    common={"run_id":out.name,"parameter_snapshot_sha256":sha(sp),"protocol_version":s["environment"]["protocol_version"]}; metrics={"schema":"amac-emotiontalk-external-metrics-v1","status":"completed",**common,"anchor":anchor,"per_seed":per_seed,"aggregate":aggregate_summary,"checks":checks}; decision={"schema":"amac-emotiontalk-external-decision-v1","status":"completed",**common,"direction_consistent":all(checks.values()),"evidence_scope":"Contract-level external validation using independent modality annotations; not end-to-end perception."}
    flds=["seed","clip_id","path","condition","gold_state","final_state","commits","prefinal_commits","prefinal_errors","revisions","premature","time_to_first","stage2_covered","final_identity"]
    with (out/"per_path.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=flds); w.writeheader(); w.writerows(all_rows)
    np.savez_compressed(out/"predictions.npz",test_ids=np.asarray([x["clip_id"] for x in test]),gold=gold,final=final,H0_risk=np.asarray(risks)); torch.save({"architecture":s["models"]["risk_predictor"],"models":models},out/"models.pt"); write(out/"metrics.json",metrics); write(out/"decision.json",decision); write(out/"costs.json",{"schema":"amac-emotiontalk-costs-v1","status":"completed",**common,"external_api_usd":0.0,"wall_seconds":time.time()-started})
    artifacts=["metrics.json","decision.json","costs.json","per_path.csv","predictions.npz","models.pt"]; write(out/"manifest.json",{"schema":"amac-emotiontalk-manifest-v1","status":"completed_unvalidated",**common,"used_groups":{"train":s["dataset"]["train_groups"],"test":s["dataset"]["test_groups"]},"counts":{"seeds":len(seeds),"test_clips":len(test),"paths":len(orders),"conditions":len(policy["conditions"]),"rows":len(all_rows)},"artifacts":{name:sha(out/name) for name in artifacts}}); print(json.dumps({"run_id":out.name,"direction_consistent":all(checks.values()),"checks":checks},ensure_ascii=False))
if __name__=="__main__": main()
