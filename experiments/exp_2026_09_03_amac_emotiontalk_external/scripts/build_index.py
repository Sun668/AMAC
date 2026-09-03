#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


MODALITIES = ("Text", "Audio", "Video", "Multimodal")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def sensor(document):
    label = document["emotion_result"]
    ratings = list(document["data"].values())
    agreeing = [item for item in ratings if item["emotion"] == label]
    confidence = sum(float(item["Confidence_degree"]) for item in agreeing) / (9.0 * len(agreeing)) if agreeing else 0.0
    return {"label": label, "agreement": len(agreeing) / len(ratings), "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--metadata-root", required=True); parser.add_argument("--output", required=True); parser.add_argument("--manifest", required=True); args = parser.parse_args()
    root, output, manifest_path = Path(args.metadata_root).resolve(), Path(args.output).resolve(), Path(args.manifest).resolve()
    if output.exists() or manifest_path.exists(): raise RuntimeError("索引或清单已存在，禁止覆盖")
    tables, source_entries = {}, []
    for modality in MODALITIES:
        table = {}
        for path in sorted((root / modality / "json").rglob("*.json")):
            relative = path.relative_to(root).as_posix(); raw_hash = digest_file(path); document = json.loads(path.read_text(encoding="utf-8"))
            key = Path(document.get("file_name") or document.get("file_path")).with_suffix("").as_posix()
            table[key] = {"speaker_id": str(document["speaker_id"]), **sensor(document)}
            source_entries.append((relative, raw_hash))
        tables[modality] = table
    common = sorted(set.intersection(*(set(table) for table in tables.values())))
    if not common: raise RuntimeError("四种模态没有可连接片段")
    rows = []
    for key in common:
        speakers = {tables[name][key]["speaker_id"] for name in MODALITIES}
        if len(speakers) != 1: raise RuntimeError(f"说话人不一致: {key}")
        rows.append({"clip_id": key, "group": key.split("/")[0], "speaker_id": speakers.pop(), "T": tables["Text"][key], "A": tables["Audio"][key], "V": tables["Video"][key], "gold": tables["Multimodal"][key]["label"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_digest = hashlib.sha256("".join(f"{path}\0{digest}\n" for path, digest in source_entries).encode()).hexdigest()
    manifest = {"schema": "emotiontalk-metadata-index-v1", "status": "completed", "source_root": str(root), "source_json_count": len(source_entries), "source_manifest_sha256": source_digest, "joined_rows": len(rows), "groups": sorted({row["group"] for row in rows}), "classes": sorted({row["gold"] for row in rows}), "gold_distribution": dict(Counter(row["gold"] for row in rows)), "index_sha256": digest_file(output), "contains_raw_content": False}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__": main()
