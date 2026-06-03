#!/usr/bin/env python3
"""
tests/test_lt_suggestions.py
=============================
Prüft die Hypothese: LanguageTool Suggestion-Objekte bei langen Listen
(SWISS_GERMAN_SPELLER_RULE) liefern kein sinnvolles str() mehr.

Aufruf:
    python tests/test_lt_suggestions.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import language_tool_python
except ImportError:
    sys.exit("[FATAL] language_tool_python nicht installiert.")


def inspect_suggestions(rule_id: str, stelle: str, replacements: list) -> None:
    print(f"\n  Rule: {rule_id} | Stelle: '{stelle}' | {len(replacements)} Vorschläge")
    for i, r in enumerate(replacements[:5]):
        attrs = {
            "type":        type(r).__name__,
            "str()":       str(r),
            ".value":      getattr(r, "value",       "–"),
            ".text":       getattr(r, "text",        "–"),
            ".replacement":getattr(r, "replacement", "–"),
        }
        print(f"    [{i}] " + " | ".join(f"{k}={v!r}" for k, v in attrs.items()))


def main():
    tool = language_tool_python.LanguageTool("de-CH")

    cases = [
        # Fall 1: Spellchecker (lange Liste erwartet)
        (
            "volständig",
            "Es entstand ein erheblicher Schaden, das Gebäude brannte volständig nieder.",
        ),
        # Fall 2: Regelbasiert, kurze Liste erwartet
        (
            "Kriminaltechnischen",
            "Unterlagen der Kriminaltechnischen Abteilung wurden gesichert.",
        ),
        # Fall 3: Weiterer Spellchecker
        (
            "befetigt",
            "Das Seil wurde befetigt und anschliessend gesichert.",
        ),
        # Fall 4: Weiterer Spellchecker
        (
            "analtischen",
            "29.	Wie beurteilen Sie den analtischen Nachweis der verbliebenen Restenergie .",
        ),

    ]

    print("="*64)
    print("  Test: LanguageTool Suggestion-Objekt-Struktur")
    print("="*64)

    for span, text in cases:
        matches = tool.check(text)
        span_matches = [
            m for m in matches
            if span in text[getattr(m,"offset",0):getattr(m,"offset",0)+getattr(m,"error_length",getattr(m,"errorLength",0))]
        ]
        if not span_matches:
            print(f"\n[WARN] '{span}' nicht erkannt.")
            continue
        for m in span_matches:
            rule_id  = str(getattr(m, "ruleId",    getattr(m, "rule_id", "")))
            offset   = int(getattr(m, "offset",    0))
            length   = int(getattr(m, "error_length", getattr(m, "errorLength", 0)))
            stelle   = text[offset:offset+length]
            reps_raw = getattr(m, "replacements", getattr(m, "suggested_replacements", []))
            inspect_suggestions(rule_id, stelle, reps_raw)

            # Zeige ob str(replacements[0]) sinnvoll ist
            if reps_raw:
                first = reps_raw[0]
                s = str(first)
                sinnvoll = "limit" not in s.lower() and "object" not in s.lower() and len(s) < 100
                print(f"    → str(replacements[0]) sinnvoll: {sinnvoll} | Wert: {s!r}")

    print("\n" + "="*64)
    print("  Hypothese: bei langen Listen (Spellchecker) ist str() unbrauchbar,")
    print("  bei kurzen Listen (regelbasiert) liefert str() den Vorschlagstext.")
    print("="*64)


if __name__ == "__main__":
    main()