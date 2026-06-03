from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_answer_reference_facts_ohneAgentUmbau import _run_spacy_adjective_check

segment = "Im Verfahren wurde die Beschuldigte Person zur Sache befragt."

findings = _run_spacy_adjective_check(segment)

print("Findings:")
for f in findings:
    print(f)

if not findings:
    raise SystemExit("FEHLER: Kein Finding erzeugt.")

if not any(
        "Beschuldigte" in f.get("stelle_im_segment", "")
        or "die Beschuldigte Person" in f.get("stelle_im_segment", "")
        for f in findings
):
    raise SystemExit("FEHLER: 'die Beschuldigte Person' wurde nicht erkannt.")

print("OK: Die Stelle wurde erkannt.")