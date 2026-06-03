# tests/print_reference_facts_case_06.py
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from importDocuments_structural import read_docx, normalize_text
from rag_answer_reference_facts_ohneAgent7 import (
    make_llm_client,
    load_reference_facts_schema,
    run_reference_facts_agent,
)

CASE_ID = "case_06"
DOC_PATH = Path("./case_documents/case_06.docx")
SCHEMA_PATH = Path("./schema/reference_facts_schema.json")
OUT_PATH = Path("./reference_facts/reference_facts_case_06_debug.json")

def main():
    llm = make_llm_client()

    doc_text = normalize_text(read_docx(DOC_PATH))
    schema = load_reference_facts_schema(SCHEMA_PATH)

    reference_facts = run_reference_facts_agent(
        llm,
        doc_text,
        case_id=CASE_ID,
        schema=schema,
        max_chars=4000,
    )

    print(json.dumps(reference_facts, ensure_ascii=False, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(reference_facts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()