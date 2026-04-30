#!/usr/bin/env python3
"""
Diagnostic: German spelling and capitalisation error detection
using spaCy (capitalisation) and hunspell (spelling/morphology).

Install:
    pip install spacy hunspell
    python -m spacy download de_dep_news_trf
    brew install hunspell   # Mac only
    # Download German dictionary files (de_DE.dic + de_DE.aff) from:
    # https://github.com/titoBouzout/Dictionaries
    # Place them in /usr/local/share/hunspell/ or adjust DIC_PATH below

Run:
    python3 tests/test_dictionnary.py
"""

import sys
import re

# ── Config ────────────────────────────────────────────────────────────────────
# pyspellchecker uses a built-in German frequency dictionary.
# Install: pip install pyspellchecker
# Note: covers common German words; rare domain terms may produce false positives.
# Add domain-specific words to DOMAIN_WHITELIST to suppress false positives.

DOMAIN_WHITELIST = {
    # Forensic / legal German terms not in the standard dictionary
    "Motorkarren", "Motorkarrens", "Motorkarrens",
    "Führerausweis", "Führerausweiskategorie",
    "Seestrasse", "Haab", "swisstopo",
    "LSI", "FOR", "KTD", "SZ",
    "Fussgänger", "Fussgängers", "Fussgängern",
    "Gutachterauftrag", "Gutachterauftrags",
    "Unfallstelle", "Unfallstellen",
    "Dumper", "Dumpers",
    "Polizeibericht", "Polizeiberichts",
}


# ── spaCy: adjective capitalisation rule ──────────────────────────────────────

def check_adjective_capitalisation(doc) -> list:
    """
    Detect wrongly capitalised attributive adjectives before nouns.
    dep_=nk is the German TIGER treebank label for noun kernel modifier.
    prev.pos_ in (DET, PUNCT, X) handles bullet-list items too.
    """
    findings = []
    for token in doc:
        if (
                token.pos_ == "ADJ"
                and token.dep_ == "nk"
                and token.head.pos_ == "NOUN"
                and token.text[0].isupper()
                and token.i > 0
                and doc[token.i - 1].pos_ in ("DET", "PUNCT", "X")
        ):
            correct = token.text[0].lower() + token.text[1:]
            findings.append({
                "tool":    "spaCy",
                "type":    "Grossschreibung Adjektiv",
                "error":   token.text,
                "correct": correct,
                "context": doc[max(0, token.i - 2): token.i + 3].text,
                "pos":     token.pos_,
                "dep":     token.dep_,
                "head":    f"{token.head.text} ({token.head.pos_})",
                "prev":    doc[token.i - 1].pos_,
            })
    return findings


# ── pyspellchecker: spelling check ───────────────────────────────────────────

def load_spellchecker():
    """
    Load pyspellchecker with German dictionary.
    Install: pip install pyspellchecker
    Pure Python, no C++ compilation needed, works on Python 3.14 / ARM Mac.
    """
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language="de", distance=1)
        # Add domain-specific terms to prevent false positives
        spell.word_frequency.load_words(DOMAIN_WHITELIST)
        print("[INFO] pyspellchecker loaded with German dictionary")
        return spell
    except ImportError:
        print("[WARN] pyspellchecker not installed — pip install pyspellchecker")
        return None
    except Exception as e:
        print(f"[WARN] pyspellchecker failed to load: {e}")
        return None


def check_spelling(text: str, spell) -> list:
    """
    Check each word against the German pyspellchecker dictionary.
    Catches spelling errors like 'Vermassungen' → 'Vermessungen'.
    Note: pyspellchecker uses frequency-based lookup, not full morphological
    analysis like hunspell, so rare genitive forms ('Brands') may not be caught.
    """
    if spell is None:
        return []

    findings = []
    for raw_word in text.split():
        # Strip surrounding punctuation
        word = re.sub(r"^[^\w]+|[^\w]+$", "", raw_word, flags=re.UNICODE)
        if not word or not any(c.isalpha() for c in word):
            continue
        # Skip proper nouns (all-caps abbreviations, words starting with uppercase
        # that are likely names — spellchecker works on lowercased input)
        if word in DOMAIN_WHITELIST:
            continue
        # Check if misspelled (pyspellchecker lowercases internally)
        misspelled = spell.unknown([word])
        if not misspelled:
            continue
        candidates = spell.candidates(word) or set()
        correction = spell.correction(word)
        findings.append({
            "tool":        "pyspellchecker",
            "type":        "Rechtschreibung",
            "error":       word,
            "correct":     correction or "?",
            "suggestions": sorted(candidates)[:3],
        })
    return findings


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── spaCy: capitalisation (SHOULD DETECT) ─────────────────────────────
    ("SHOULD DETECT (cap)",  "Die Beschuldigte Person fuhr zu schnell."),
    ("SHOULD DETECT (cap)",  "Der Beschuldigte Fahrer hat das Steuer verloren."),
    ("SHOULD DETECT (cap)",  "Das Beschuldigte Fahrzeug war auf der Seestrasse."),
    ("SHOULD DETECT (cap)",  "Die Zeugin sah die Beschuldigte Person am Unfallort."),
    ("SHOULD DETECT (cap)",  "Beigezogene Videoaufnahmen des Unfalls wurden analysiert."),
    ("SHOULD DETECT (cap)",  "Zusätzliche Unterlagen wurden beigezogen."),
    ("SHOULD DETECT (cap)",  "die Beschuldigte Person hatte keine gültige Führerausweis-Kategorie."),

    # ── spaCy: capitalisation (SHOULD IGNORE) ─────────────────────────────
    ("SHOULD IGNORE (cap)",  "Die beschuldigte Person fuhr zu schnell."),
    ("SHOULD IGNORE (cap)",  "Die Beschuldigte wurde befragt."),
    ("SHOULD IGNORE (cap)",  "Die Aussage der Beschuldigten war widersprüchlich."),

    # ── hunspell: spelling / morphology (SHOULD DETECT) ──────────────────
    ("SHOULD DETECT (spell)", "Im ersten Schritt wurde der angebliche Ausgangsort des Brands untersucht."),
    ("SHOULD DETECT (spell)", "Die Vermassungen wurden durch das Forensische Institut erstellt."),
    ("SHOULD DETECT (spell)", "Das Gutachten LSI geht von einer Gluttemperatur der Holzkohle von 473 Grad aus."),

    # ── hunspell: spelling (SHOULD IGNORE — correct forms) ───────────────
    ("SHOULD IGNORE (spell)", "Im ersten Schritt wurde der angebliche Ausgangsort des Brandes untersucht."),
    ("SHOULD IGNORE (spell)", "Die Vermessungen wurden durch das Forensische Institut erstellt."),

    # ── Neither tool should flag these ────────────────────────────────────
    ("OTHER",  "- Haben Sie weitere sachdienliche Hinweise? -"),
    ("OTHER",  "Höhe Beginn zweites Parkfeld"),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading spaCy model de_dep_news_trf...")
    try:
        import spacy
        nlp = spacy.load("de_dep_news_trf")
        print(f"Model loaded: {nlp.meta['name']} v{nlp.meta['version']}")
    except OSError:
        print("ERROR: Model not found — python -m spacy download de_dep_news_trf")
        sys.exit(1)
    except ImportError:
        print("ERROR: spaCy not installed — pip install spacy")
        sys.exit(1)

    spell = load_spellchecker()
    print()

    print("=" * 70)
    for expected, sentence in TEST_CASES:
        doc = nlp(sentence)

        cap_findings   = check_adjective_capitalisation(doc)
        spell_findings = check_spelling(sentence, spell)
        all_findings   = cap_findings + spell_findings

        print(f"[{expected}]")
        print(f"  INPUT : {sentence!r}")

        if "cap" in expected.lower():
            token_info = [(t.text, t.pos_, t.dep_, t.head.text) for t in doc]
            print(f"  TOKENS: {token_info}")

        if all_findings:
            for f in all_findings:
                tool = f["tool"]
                if tool == "spaCy":
                    print(f"  [{tool}] FOUND : '{f['error']}' → '{f['correct']}'")
                    print(f"           context={f['context']!r} prev_pos={f['prev']} head={f['head']}")
                else:
                    print(f"  [{tool}] FOUND : '{f['error']}' → '{f['correct']}'")
                    print(f"           suggestions={f['suggestions']}")
        else:
            print("  → No errors detected")
        print()

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()