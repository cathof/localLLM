# tests/test_spacy_beschuldigte.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_answer_reference_facts_ohneAgent7 import _run_spacy_adjective_check

text = "die Beschuldigte Person"

findings = _run_spacy_adjective_check(text)
print(findings)