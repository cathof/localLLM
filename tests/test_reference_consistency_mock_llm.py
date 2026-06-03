from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_answer_reference_facts_OHNEFIX import (
    run_reference_consistency_agent,
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
    def allowed_subclasses_by_main_label(self):
        return {
            "Struktur und Argumentation": {
                "Beschreibung von Befunden",
                "Evidenz / Belege",
            },
            "Rechtskonformität": {
                "Aussageabsicherung",
                "Trennung Befund/Bewertung",
            },
            "Formales": {
                "Redaktionelle Korrektur",
            },
        }

    @property
    def allowed_change_labels(self):
        return {
            "Fachliche Präzisierung",
            "Redaktionelle Korrektur",
            "Rechnerische Korrektur",
            "Hypothesen-Korrektur",
        }

    @property
    def allowed_severity_labels(self):
        return {"niedrig", "mittel", "hoch"}

    @property
    def subclass_label_to_main_label(self):
        return {
            "Beschreibung von Befunden": "Struktur und Argumentation",
            "Evidenz / Belege": "Struktur und Argumentation",
            "Aussageabsicherung": "Rechtskonformität",
            "Trennung Befund/Bewertung": "Rechtskonformität",
            "Redaktionelle Korrektur": "Formales",
        }


class MockLLM:
    def chat(self, messages, json_mode=False, schema=None):
        # Optional: kurz prüfen, dass der Prompt wirklich den Referenzfakt und die Dokumentstelle enthält.
        joined = "\n\n".join(m["content"] for m in messages)
        if "Staatsanwaltschaft Schwyz" not in joined:
            raise RuntimeError("Prompt enthält den Referenz-Auftraggeber nicht.")
        if "Jugendanwaltschaft Obwalden" not in joined:
            raise RuntimeError("Prompt enthält die abweichende Dokumentstelle nicht.")

        return json.dumps(
            {
                "errors": [
                    {
                        "hauptklasse": "Struktur und Argumentation",
                        "subklasse": "Beschreibung von Befunden",
                        "aenderungstyp": "Fachliche Präzisierung",
                        "schweregrad": "hoch",
                        "reference_key": "auftraggeber",
                        "reference_value": "Staatsanwaltschaft Schwyz",
                        "stelle_im_segment": (
                            "Jugendanwältin Frau MLaw Viviane Quadri, "
                            "Jugendanwaltschaft Obwalden"
                        ),
                        "begruendung": (
                            "Widerspruch: Der Referenzfakt definiert den Auftraggeber "
                            "als Staatsanwaltschaft Schwyz; im Dokument wird die "
                            "Auftragserteilung der Jugendanwaltschaft Obwalden bzw. "
                            "Jugendanwältin Viviane Quadri zugeschrieben."
                        ),
                        "vorschlag": "Staatsanwaltschaft Schwyz",
                    }
                ]
            },
            ensure_ascii=False,
        )


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

segment = (
    "Mit Schreiben vom 20.03.2025 erteilte Jugendanwältin Frau MLaw Viviane Quadri, "
    "Jugendanwaltschaft Obwalden, der sachverständigen Person Andreas Leu, "
    "Forensisches Institut Zürich (FOR), den Auftrag."
)

doc_text = segment
segments = [segment]

findings = run_reference_consistency_agent(
    llm=MockLLM(),
    doc_text=doc_text,
    segments=segments,
    reference_facts=reference_facts,
    catalog=DummyCatalog(),
)

print("FINDINGS:")
print(json.dumps(findings, ensure_ascii=False, indent=2))

if len(findings) != 1:
    raise SystemExit(f"FEHLER: Erwartet wurde genau 1 Finding, erhalten: {len(findings)}")

finding = findings[0]

expected = {
    "segment_index": 1,
    "hauptklasse": "Struktur und Argumentation",
    "subklasse": "Beschreibung von Befunden",
    "aenderungstyp": "Fachliche Präzisierung",
    "schweregrad": "hoch",
    "reference_key": "auftraggeber",
    "reference_value": "Staatsanwaltschaft Schwyz",
    "stelle_im_segment": "Jugendanwältin Frau MLaw Viviane Quadri, Jugendanwaltschaft Obwalden",
    "vorschlag": "Staatsanwaltschaft Schwyz",
}

for key, value in expected.items():
    if finding.get(key) != value:
        raise SystemExit(
            f"FEHLER bei {key}: erwartet {value!r}, erhalten {finding.get(key)!r}"
        )

if finding.get("source_refs") != ["DOC_INTERNAL"]:
    raise SystemExit(
        f"FEHLER: source_refs sollte ['DOC_INTERNAL'] sein, erhalten: {finding.get('source_refs')!r}"
    )

print("\nOK: Mock-LLM-Finding wird von Agent 6 korrekt normalisiert und Segment 1 zugeordnet.")