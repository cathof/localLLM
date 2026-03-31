import json
from pathlib import Path

target = "AWE-30024.docx"   # Beispiel
path = Path("prepared.jsonl")

rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        meta = obj.get("metadata", {})
        if meta.get("source_name") == target:
            rows.append(obj)

rows.sort(key=lambda x: x["metadata"].get("chunk_index", -1))

for r in rows:
    meta = r["metadata"]
    print("=" * 100)
    print("chunk_index:", meta.get("chunk_index"))
    print("section_title:", meta.get("section_title"))
    print("chunk_len:", meta.get("chunk_len"))
    print(r["text"][:1500])