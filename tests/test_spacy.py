#!/usr/bin/env python3
"""
Diagnostic: does spaCy de_dep_news_trf detect German capitalisation errors
in forensic legal documents?

Install:
    pip install spacy
    python -m spacy download de_dep_news_trf

Run:
    python3 tests/test_spacy.py
"""

import sys


def check_adjective_capitalisation(doc) -> list:
    findings = []
    for token in doc:
        if (
                token.pos_ == "ADJ"
                and token.dep_ == "nk"
                and token.head.pos_ == "NOUN"
                and token.text[0].isupper()
                and token.i > 0
                and doc[token.i - 1].pos_ == "DET"
        ):
            correct = token.text[0].lower() + token.text[1:]
            findings.append({
                "type":    "Grossschreibung Adjektiv",
                "error":   token.text,
                "correct": correct,
                "context": doc[max(0, token.i - 2): token.i + 3].text,
                "pos":     token.pos_,
                "dep":     token.dep_,
                "head":    f"{token.head.text} ({token.head.pos_})",
            })
    return findings


TEST_CASES = [
    # ── Should be detected (adjective before noun, wrongly capitalised) ──────
    ("SHOULD DETECT",  "Die Beschuldigte Person fuhr zu schnell."),
    ("SHOULD DETECT",  "Der Beschuldigte Fahrer hat das Steuer verloren."),
    ("SHOULD DETECT",  "Das Beschuldigte Fahrzeug war auf der Seestrasse."),
    ("SHOULD DETECT",  "Die Zeugin sah die Beschuldigte Person am Unfallort."),
    ("SHOULD DETECT",  "beiden am Unfallbeteiligten Fussgänger überquerten die Strasse."),
    # ── Should NOT be detected (correct lowercase) ───────────────────────────
    ("SHOULD IGNORE",  "Die beschuldigte Person fuhr zu schnell."),
    ("SHOULD IGNORE",  "Der beschuldigte Fahrer hat das Steuer verloren."),
    # ── Should NOT be detected (nominalised — 'die Beschuldigte' is a noun) ──
    ("SHOULD IGNORE",  "Die Beschuldigte wurde befragt."),
    ("SHOULD IGNORE",  "Die Aussage der Beschuldigten war widersprüchlich."),
    # ── Other error types from ground truth ──────────────────────────────────
    ("OTHER",          "- Haben Sie weitere sachdienliche Hinweise? -"),
    ("OTHER",          "Höhe Beginn zweites Parkfeld"),
    ("OTHER",          "Einvernahmeprotokoll vom Thomas Müller"),
    ("OTHER",          "2880 x 1660 Pixel"),
    # ── Full sentences from the actual document ───────────────────────────────
    ("REAL DOC",       "die Beschuldigte Person hatte keine gültige Führerausweis-Kategorie."),
    ("REAL DOC",       "Beide am Unfall beteiligten Fussgänger wurden verletzt."),
]


def main():
    print("Loading spaCy model de_dep_news_trf...")
    try:
        import spacy
        nlp = spacy.load("de_dep_news_trf")
        print(f"Model loaded: {nlp.meta['name']} v{nlp.meta['version']}\n")
    except OSError:
        print("ERROR: Model not found. Run:")
        print("  python -m spacy download de_dep_news_trf")
        sys.exit(1)
    except ImportError:
        print("ERROR: spaCy not installed. Run:")
        print("  pip install spacy")
        sys.exit(1)

    print("=" * 70)
    for expected, sentence in TEST_CASES:
        doc = nlp(sentence)
        findings = check_adjective_capitalisation(doc)

        print(f"[{expected}]")
        print(f"  INPUT : {sentence!r}")

        # Also show the full token analysis for insight
        token_info = [(t.text, t.pos_, t.dep_, t.head.text) for t in doc]
        print(f"  TOKENS: {token_info}")

        if findings:
            for f in findings:
                print(f"  FOUND : '{f['error']}' → '{f['correct']}'")
                print(f"          context={f['context']!r} pos={f['pos']} dep={f['dep']} head={f['head']}")
        else:
            print("  → No adjective capitalisation errors detected")
        print()

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()