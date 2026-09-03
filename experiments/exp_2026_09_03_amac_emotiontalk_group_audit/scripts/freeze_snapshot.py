#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--template", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]; config = json.loads(Path(args.template).read_text()); config["status"] = "frozen"
    for key in ("rows", "validator", "snapshot"): config["models"][f"{key}_sha256"] = sha256(root / config["models"][key])
    for key in ("runner", "validator"): config["environment"][f"{key}_sha256"] = sha256(root / config["environment"][f"{key}_path"])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "frozen", "path": str(output.resolve()), "sha256": sha256(output)}))

if __name__ == "__main__": main()
