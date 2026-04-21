import argparse
import json
import unicodedata
from pathlib import Path


def normalize_for_match(s: str) -> str:
    """
    Normalize Unicode differences (e.g. ä vs a + combining ¨),
    collapse whitespace, and lowercase for robust filename matching.
    """
    s = unicodedata.normalize("NFC", s or "")
    s = " ".join(s.split())
    return s.lower().strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect chunks for one source document in a prepared JSONL file."
    )
    ap.add_argument("--prepared", type=str, default="prepared_rules.jsonl")
    ap.add_argument("--source_name", type=str, required=True)
    ap.add_argument("--case_id", type=str, default="")
    ap.add_argument("--document_type", type=str, default="")
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument(
        "--exact",
        action="store_true",
        help="Require exact normalized match of source_name. Default: substring match.",
    )
    args = ap.parse_args()

    path = Path(args.prepared)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    query_source_name = normalize_for_match(args.source_name)
    query_case_id = normalize_for_match(args.case_id) if args.case_id else ""
    query_document_type = normalize_for_match(args.document_type) if args.document_type else ""

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("metadata", {})

            meta_source_name = normalize_for_match(str(meta.get("source_name", "")))
            meta_case_id = normalize_for_match(str(meta.get("case_id", "")))
            meta_document_type = normalize_for_match(str(meta.get("document_type", "")))

            if args.exact:
                if meta_source_name != query_source_name:
                    continue
            else:
                if query_source_name not in meta_source_name:
                    continue

            if query_case_id and meta_case_id != query_case_id:
                continue
            if query_document_type and meta_document_type != query_document_type:
                continue

            rows.append(obj)

    rows.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", -1))

    if not rows:
        print("No matching chunks found.")
        return

    for r in rows:
        meta = r["metadata"]
        print("=" * 100)
        print("source_name:", meta.get("source_name"))
        print("source_path:", meta.get("source_path"))
        print("case_id:", meta.get("case_id"))
        print("document_type:", meta.get("document_type"))
        print("chunk_index:", meta.get("chunk_index"))
        print("section_title:", meta.get("section_title"))
        print("chunk_len:", meta.get("chunk_len"))
        print(r["text"][:args.max_chars])


if __name__ == "__main__":
    main()