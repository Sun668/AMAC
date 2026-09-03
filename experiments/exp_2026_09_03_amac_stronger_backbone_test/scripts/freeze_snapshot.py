#!/usr/bin/env python3
import argparse, hashlib, json, platform
from pathlib import Path
import numpy, sklearn, torch

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--template", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]; output = Path(args.output).resolve(); value = json.loads(Path(args.template).read_text(encoding="utf-8"))
    if output.exists(): raise RuntimeError("冻结快照已存在，禁止覆盖")
    if value["status"] != "draft": raise RuntimeError("模板状态必须为 draft")
    dataset = root / value["dataset"]["path"]
    if sha256(dataset) != value["dataset"]["sha256"]: raise RuntimeError("数据集哈希不一致")
    for key in ("runner", "validator", "development_runner", "base_runner", "matrix_runner", "governor_profile"):
        value["environment"][f"{key}_sha256"] = sha256(root / value["environment"][f"{key}_path"])
    for key in ("decision", "validator"):
        path = root / value["development"][key]; value["development"][f"{key}_sha256"] = sha256(path)
    reference = root / value["models"]["control"]["artifact"]; value["models"]["control"]["artifact_sha256"] = sha256(reference)
    value["environment"].update({"python": platform.python_version(), "numpy": numpy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}); value["status"] = "frozen"
    output.parent.mkdir(parents=True, exist_ok=False); output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "path": str(output), "sha256": sha256(output)}, ensure_ascii=False))

if __name__ == "__main__": main()

