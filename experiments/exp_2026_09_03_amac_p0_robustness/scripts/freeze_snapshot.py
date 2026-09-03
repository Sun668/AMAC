#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    experiment = Path(__file__).resolve().parents[1]
    output = experiment / "snapshots" / args.run_id / "parameters.json"
    if output.exists():
        raise RuntimeError("冻结参数快照已存在，禁止覆盖")
    value = json.loads((experiment / "parameters.template.json").read_text(encoding="utf-8"))
    value.update({"status": "frozen", "run_id": args.run_id, "frozen_at": datetime.now(timezone.utc).isoformat()})
    value["artifacts"] = {
        key: {"path": path, "sha256": sha256(root / path)}
        for key, path in value["artifacts"].items()
    }
    for key, name in (("runner", "run_analysis.py"), ("validator", "validate_analysis.py"), ("paper_renderer", "render_paper_tables.py")):
        path = str((experiment / "scripts" / name).relative_to(root))
        value["environment"][f"{key}_path"] = path
        value["environment"][f"{key}_sha256"] = sha256(root / path)
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "snapshot": str(output), "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
