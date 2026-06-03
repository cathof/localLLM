#!/usr/bin/env python3
"""
tests/test_language_tool.py
============================
Testet ob LanguageTool de-CH einen injizierten Fehler erkennt –
exakt so wie es in rag_answer_reference_facts.py aufgerufen wird.

Aufruf:
    python tests/test_language_tool.py
    python tests/test_language_tool.py --span "volständig" --segment "...text..."
"""

import argparse
import sys
from pathlib import Path

# Root-Verzeichnis zum Pfad hinzufügen (Script liegt in tests/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rag_answer_reference_facts_OHNEFIX import (
        load_dotenv,
        _get_language_tool,
        _run_language_tool,
        _LT_ALLOWED_RULE_PREFIXES,
        _LT_BLOCKED_RULE_IDS,
        load_taxonomy_json,
        env_str,
    )
except ImportError as e:
    sys.exit(f"[FATAL] Import fehlgeschlagen: {e}")

load_dotenv(".env")


def test_span(span: str, segment_text: str, taxonomy_path: Path) -> None:
    """
    Testet ob LanguageTool den gegebenen Span im Segment erkennt.
    Zeigt alle gefundenen Matches und ob sie durch den Rule-Filter kommen.
    """
    sep = "─" * 60

    print(f"\n{sep}")
    print(f"Span:    '{span}'")
    print(f"Segment: '{segment_text[:100]}...' " if len(segment_text) > 100 else f"Segment: '{segment_text}'")
    print(sep)

    # ── 1. LanguageTool initialisieren ───────────────────────────────────────
    tool = _get_language_tool()
    if tool is None:
        print("[FAIL] LanguageTool nicht verfügbar.")
        return
    print("[OK] LanguageTool de-CH geladen.")

    # ── 2. Rohe Matches (vor Filter) ─────────────────────────────────────────
    try:
        matches = tool.check(segment_text)
    except Exception as e:
        print(f"[FAIL] tool.check() Fehler: {e}")
        return

    print(f"\n── Alle Matches (vor Rule-Filter): {len(matches)} ──")
    span_found_raw = False
    for m in matches:
        offset  = int(getattr(m, "offset",      getattr(m, "offsetInContext", 0)))
        length  = int(getattr(m, "errorLength", getattr(m, "length", 0)))
        rule_id = str(getattr(m, "ruleId",      getattr(m, "rule_id", "")))
        message = str(getattr(m, "message",     getattr(m, "msg", "")))
        stelle  = segment_text[offset:offset + length].strip() if length > 0 else ""
        reps    = getattr(m, "replacements", getattr(m, "suggested_replacements", []))
        first_rep = str(reps[0]) if reps else "–"

        is_span = span in stelle or stelle in span
        marker  = " ◄ ZIEL-SPAN" if is_span else ""
        if is_span:
            span_found_raw = True

        # Rule-Filter prüfen
        blocked  = rule_id in _LT_BLOCKED_RULE_IDS
        allowed  = (not rule_id) or any(rule_id.startswith(p) for p in _LT_ALLOWED_RULE_PREFIXES)
        passes   = not blocked and allowed
        filter_status = "[PASS]" if passes else "[FILTERED]"

        print(f"  {filter_status} ruleId={rule_id:<35} stelle='{stelle}'  → '{first_rep}'{marker}")
        if is_span:
            print(f"           message: {message}")

    if not span_found_raw:
        print(f"\n[FAIL] Span '{span}' wurde von LanguageTool gar nicht gefunden.")
        print(f"       → LanguageTool kennt '{span}' nicht als Fehler in de-CH.")

    # ── 3. Gefilterte Findings (wie im Produktionscode) ───────────────────────
    catalog = load_taxonomy_json(Path(env_str("TAXONOMY_JSON", "tax/taxonomy.json")))
    findings = _run_language_tool(segment_text, catalog)

    print(f"\n── Findings nach Rule-Filter: {len(findings)} ──")
    span_found_filtered = False
    for f in findings:
        stelle = f.get("stelle_im_segment", "")
        is_span = span in stelle or stelle in span
        if is_span:
            span_found_filtered = True
        marker = " ◄ ZIEL-SPAN" if is_span else ""
        print(f"  stelle='{stelle}'  subklasse={f.get('subklasse','')}  "
              f"vorschlag='{f.get('vorschlag','')}'  score={f.get('konfidenz','')} {marker}")

    if not findings:
        print("  (keine Findings)")

    # ── 4. Fazit ─────────────────────────────────────────────────────────────
    print(f"\n── Fazit ──")
    if span_found_filtered:
        print(f"[OK] '{span}' erkannt und durch Rule-Filter gekommen.")
        print(f"     → Agent 3 würde diesen Fehler melden.")
    elif span_found_raw:
        print(f"[WARN] '{span}' von LanguageTool erkannt, aber durch Rule-Filter entfernt.")
        print(f"       → Die Rule-ID ist nicht in _LT_ALLOWED_RULE_PREFIXES.")
        print(f"       → Lösung: entsprechende Rule-Familie erlauben oder Filter anpassen.")
    else:
        print(f"[FAIL] '{span}' nicht erkannt.")
        print(f"       → LanguageTool de-CH kennt diesen Tippfehler nicht.")
        print(f"       → Solche Fehler sind für Agent 3 nicht erkennbar.")
        print(f"       → Im Injector: diesen Subtyp vorsichtiger einsetzen.")

    print()


def main():
    ap = argparse.ArgumentParser(description="LanguageTool Fehlererkennungstest")
    ap.add_argument(
        "--span",
        default="volständig",
        help="Der injizierte Span (Tippfehler), der erkannt werden soll"
    )
    ap.add_argument(
        "--segment",
        default=(
            "Es entstand ein erheblicher Schaden an Wald und Wiesland sowie ein im "
            "Brandgebiet stehendes Gebäude, welches volständig niederbrannte. "
            "Die Brandursachenuntersuchung und Spurensicherung vor Ort erfolgte durch "
            "die Kantonspolizei Wallis in Zusammenarbeit mit dem Forensischen Institut Zürich."
        ),
        help="Segment-Text in dem der Span gesucht wird"
    )
    args = ap.parse_args()

    # Span im Segment prüfen
    if args.span not in args.segment:
        print(f"[WARN] '{args.span}' nicht im Segment-Text — teste trotzdem.")

    taxonomy_path = Path(env_str("TAXONOMY_JSON", "tax/taxonomy.json"))
    test_span(args.span, args.segment, taxonomy_path)

    # ── Weitere typische Testfälle ────────────────────────────────────────────
    if args.span == "volständig":
        print("="*60)
        print("Zusätzliche Testfälle aus dem synthetischen Fehlerkorpus:")
        print("="*60)

        cases = [
            ("befetigt",    "Das Seil wurde befetigt und gesichert."),
            ("Erkennbare",  "Erkennbare Brandspuren zeigten sich an der Wand."),
            ("Einiger",     "Seit Einiger Zeit herrschte Trockenheit in der Region."),
            ("analtischen", "Die analtischen Ergebnisse wurden ausgewertet."),
            ("Late",        "Die unterste bergwärts laufende Late war beschädigt."),
        ]

        for span, seg in cases:
            tool = _get_language_tool()
            if tool is None:
                break
            matches = tool.check(seg)
            found = any(
                span in seg[getattr(m, "offset", 0):getattr(m, "offset", 0) + getattr(m, "errorLength", 0)]
                for m in matches
            )
            status = "[OK]  " if found else "[FAIL]"
            print(f"  {status} '{span}' in: '{seg[:60]}...'")


if __name__ == "__main__":
    main()