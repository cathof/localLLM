# tests/test_legal_abbrev_guard.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_answer_reference_facts_ohneAgent7 import check_legal_abbreviation_variants

TESTS = [
    "Auszug StPo und StGB",
    "Auszug Stpo und StGB",
    "Gemäss Art. 3 JStpo i.V.m. Art. 182 ff. StPO",
    "Dies steht ivm Art. 3 StGB.",
    "Siehe aaO.",
]

def main():
    for i, text in enumerate(TESTS, start=1):
        findings = check_legal_abbreviation_variants(text, i)
        print("=" * 80)
        print(text)
        print(findings)

if __name__ == "__main__":
    main()