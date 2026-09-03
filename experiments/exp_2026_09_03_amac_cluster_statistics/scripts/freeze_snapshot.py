#!/usr/bin/env python3
import argparse, hashlib, json, platform
from pathlib import Path
import numpy
import scipy

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--template", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]; output = Path(args.output).resolve(); value = json.loads(Path(args.template).read_text(encoding="utf-8"))
    if output.exists(): raise RuntimeError("冻结快照已存在，禁止覆盖")
    for model in ("ridge", "masked_fusion"):
        for kind in ("rows", "validator", "snapshot"):
            value["models"][model][f"{kind}_sha256"] = sha256(root / value["models"][model][kind])
        report = json.loads((root / value["models"][model]["validator"]).read_text())
        if report.get("status") != "passed": raise RuntimeError(f"{model} 上游验证未通过")
    for kind in ("runner", "validator", "source_audit"):
        value["environment"][f"{kind}_sha256"] = sha256(root / value["environment"][f"{kind}_path"])
    value["environment"].update({"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}); value["status"] = "frozen"
    output.parent.mkdir(parents=True, exist_ok=False); output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "path": str(output), "sha256": sha256(output)}, ensure_ascii=False))

if __name__ == "__main__": main()

