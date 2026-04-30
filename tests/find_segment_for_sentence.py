# tests/find_segment_for_sentence.py

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from importDocuments_structural import read_docx, normalize_text
from rag_answer_multi_query_diverse_rewritten_new import split_document_into_segments


def norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True, help="Pfad zum .docx Dokument")
    ap.add_argument("--sentence", required=True, help="Satz oder Textstelle")
    args = ap.parse_args()

    doc_path = Path(args.document).expanduser().resolve()

    doc_text = normalize_text(read_docx(doc_path))
    segments = split_document_into_segments(doc_text)

    found = False
    for i, segment in enumerate(segments, start=1):
        if norm(args.sentence) in norm(segment):
            print("=" * 90)
            print(f"GEFUNDEN IN SEGMENT {i}")
            print("=" * 90)
            print(segment)
            found = True

    if not found:
        print(f"Nicht gefunden: {args.sentence!r}")
        print(f"Anzahl Segmente: {len(segments)}")


if __name__ == "__main__":
    main()