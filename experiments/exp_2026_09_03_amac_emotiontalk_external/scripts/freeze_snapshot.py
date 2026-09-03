#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while c:=f.read(8*1024*1024): h.update(c)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--template",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    root=Path(__file__).resolve().parents[3]; out=Path(a.output).resolve()
    if out.exists(): raise RuntimeError("冻结快照已存在，禁止覆盖")
    d=json.loads(Path(a.template).read_text())
    for key in ("runner","validator","index","index_manifest","development_runner"):
        d["environment"][f"{key}_sha256"]=sha(root/d["environment"][f"{key}_path"])
    d["status"]="frozen"; out.parent.mkdir(parents=True,exist_ok=False); out.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"snapshot":str(out),"sha256":sha(out)},ensure_ascii=False))
if __name__=="__main__": main()
