from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_answer_reference_facts_OHNEFIX import (
    _compact_reference_facts_for_consistency,
    build_reference_consistency_review_messages,
)


class DummyCatalog:
    @property
    def allowed_main_labels(self):
        return {
            "Struktur und Argumentation",
            "Rechtskonformität",
            "Formales",
            "Rechenfehler",
            "Hypothesenprüfung",
        }

    @property
    def allowed_change_labels(self):
        return {
            "Fachliche Präzisierung",
            "Redaktionelle Korrektur",
            "Rechnerische Korrektur",
            "Hypothesen-Korrektur",
        }


reference_facts = {
    "case_id": "case_06",
    "facts": {
        "auftraggeber": {
            "value": "Staatsanwaltschaft Schwyz",
            "source_span": "Staatsanwaltschaft Schwyz",
            "confidence": "high",
        },
        "personen": [
            {
                "name": "Frau JUGAin MLaw Viviane Quadri",
                "rolle": "",
                "source_span": "Frau JUGAin MLaw Viviane Quadri",
                "confidence": "medium",
            }
        ],
    },
}

doc_text = (
    "Mit Schreiben vom 20.03.2025 erteilte Jugendanwältin Frau MLaw Viviane Quadri, "
    "Jugendanwaltschaft Obwalden, der sachverständigen Person Andreas Leu, "
    "Forensisches Institut Zürich (FOR), den Auftrag."
)

compact = _compact_reference_facts_for_consistency(reference_facts)

print("COMPACT REFERENCE FACTS:")
print(compact)

if compact["facts"]["auftraggeber"]["value"] != "Staatsanwaltschaft Schwyz":
    raise SystemExit("FEHLER: Auftraggeber wurde nicht korrekt übernommen.")

if compact["facts"]["personen"][0]["name"] != "Frau JUGAin MLaw Viviane Quadri":
    raise SystemExit("FEHLER: Person wurde nicht korrekt übernommen.")

messages = build_reference_consistency_review_messages(
    doc_text=doc_text,
    reference_facts=reference_facts,
    catalog=DummyCatalog(),
)

system_prompt = messages[0]["content"]
user_prompt = messages[1]["content"]

checks = [
    ("Auftraggeber-Konsistenz", system_prompt),
    ("erteilte ... den Auftrag", system_prompt),
    ("andere Behörde, Institution oder Person", system_prompt),
    ("Staatsanwaltschaft Schwyz", user_prompt),
    ("Jugendanwaltschaft Obwalden", user_prompt),
]

for needle, haystack in checks:
    if needle not in haystack:
        raise SystemExit(f"FEHLER: Erwarteter Text fehlt im Prompt: {needle}")

print("\nOK: Auftraggeber-Referenzfakt wird behalten und Agent-6-Prompt enthält die Rollenregel.")