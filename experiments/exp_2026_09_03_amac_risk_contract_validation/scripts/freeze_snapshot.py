#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    template = Path(args.template).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("冻结快照已存在，禁止覆盖")
    document = json.loads(template.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[3]
    document["status"] = "frozen"
    document["environment"]["runner_sha256"] = sha256(root / document["environment"]["runner_path"])
    document["environment"]["validator_sha256"] = sha256(root / document["environment"]["validator_path"])
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
