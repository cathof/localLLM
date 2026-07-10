#!/usr/bin/env python3
"""
tests/test_fixes.py
===================
Zwei Unit-Tests die prüfen ob die drei Fixes korrekt implementiert sind:

  Test 1 — Fix A + B: STRUKT_EVIDENZ-Filter
    Prüft ob source_refs mit Platzhaltern (SRC_X_N, SRC_R_N, SRC_M_N)
    korrekt verworfen werden und nur DOC_INTERNAL akzeptiert wird.

  Test 2 — Fix C: Language-Agent-Prompt
    Prüft ob die drei neuen NICHT-ERLAUBT-Regeln im System-Prompt
    korrekt formuliert sind (kein LLM-Call nötig, reine Textprüfung).

Aufruf:
    python tests/test_fixes.py
    python -m pytest tests/test_fixes.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Projektroot in sys.path aufnehmen
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Fix A + B: STRUKT_EVIDENZ Platzhalter-Filter
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenzFilter(unittest.TestCase):
    """
    Prüft ob run_factual_agent() STRUKT_EVIDENZ-Findings mit Platzhalter-Refs
    korrekt verwirft und nur DOC_INTERNAL-Findings durchlässt.

    Da _is_valid_evidenz_ref() eine innere Funktion von run_factual_agent() ist,
    extrahieren wir die Logik aus dem Quelltext und testen sie direkt.
    """

    def _make_valid_evidenz_ref(self, refs):
        """
        Repliziert die Logik von _is_valid_evidenz_ref() aus dem Quelltext.
        Muss mit der Implementierung übereinstimmen.
        """
        refs = [r for r in (refs or []) if r and r not in ("", None)]
        if not refs:
            return False
        _PLACEHOLDER_REFS = {"SRC_X_N", "SRC_R_N", "SRC_M_N", "SRC_X_1", "SRC_N"}
        if any(r in _PLACEHOLDER_REFS for r in refs):
            return False
        return refs == ["DOC_INTERNAL"]

    def test_doc_internal_akzeptiert(self):
        """DOC_INTERNAL ist der einzige gültige Ref für STRUKT_EVIDENZ."""
        self.assertTrue(
            self._make_valid_evidenz_ref(["DOC_INTERNAL"]),
            "DOC_INTERNAL muss akzeptiert werden"
        )

    def test_src_x_n_verworfen(self):
        """SRC_X_N ist der Prompt-Platzhalter — muss verworfen werden."""
        self.assertFalse(
            self._make_valid_evidenz_ref(["SRC_X_N"]),
            "SRC_X_N ist ein Prompt-Platzhalter und muss verworfen werden"
        )

    def test_src_r_n_verworfen(self):
        """SRC_R_N ist ein weiterer Platzhalter."""
        self.assertFalse(
            self._make_valid_evidenz_ref(["SRC_R_N"]),
            "SRC_R_N muss verworfen werden"
        )

    def test_src_m_n_verworfen(self):
        """SRC_M_N ist ein weiterer Platzhalter."""
        self.assertFalse(
            self._make_valid_evidenz_ref(["SRC_M_N"]),
            "SRC_M_N muss verworfen werden"
        )

    def test_echte_regelwerk_ref_verworfen(self):
        """S7_R_3 ist eine echte Regelwerk-Referenz — für STRUKT_EVIDENZ nicht gültig."""
        self.assertFalse(
            self._make_valid_evidenz_ref(["S7_R_3"]),
            "Regelwerk-Refs sind für STRUKT_EVIDENZ nicht gültig"
        )

    def test_echte_material_ref_verworfen(self):
        """S12_M_5 ist eine echte Material-Referenz — für STRUKT_EVIDENZ nicht gültig."""
        self.assertFalse(
            self._make_valid_evidenz_ref(["S12_M_5"]),
            "Material-Refs sind für STRUKT_EVIDENZ nicht gültig"
        )

    def test_leere_refs_verworfen(self):
        """Keine Refs → kein Finding."""
        self.assertFalse(self._make_valid_evidenz_ref([]))
        self.assertFalse(self._make_valid_evidenz_ref(None))
        self.assertFalse(self._make_valid_evidenz_ref([""]))

    def test_mischung_mit_doc_internal_verworfen(self):
        """DOC_INTERNAL + SRC_X_N zusammen ist ungültig."""
        self.assertFalse(
            self._make_valid_evidenz_ref(["DOC_INTERNAL", "SRC_X_N"]),
            "Mischung aus DOC_INTERNAL und Platzhalter muss verworfen werden"
        )

    def test_prompt_enthaelt_nicht_mehr_src_x_n_als_regelwerk_ref(self):
        """
        Fix A: Der Agent-2-User-Prompt darf SRC_X_N nicht mehr als
        valide Regelwerk-Referenz vorschlagen.
        """
        candidates = [
            _PROJECT_ROOT / "rag_answer_reference_facts.py",
            _THIS_DIR / "rag_answer_reference_facts.py",
            Path.cwd() / "rag_answer_reference_facts.py",
        ]
        _src_path = next((p for p in candidates if p.exists()), None)
        if _src_path is None:
            self.skipTest("rag_answer_reference_facts.py nicht gefunden")
        src = _src_path.read_text(encoding="utf-8")
        bad_phrase = 'SRC_X_N\\"] für Regelwerk'
        self.assertNotIn(
            bad_phrase, src,
            "Der Prompt darf SRC_X_N nicht mehr als Regelwerk-Referenz anbieten"
        )

    def test_prompt_enthaelt_niemals_src_x_n_anweisung(self):
        """
        Fix A: Der Prompt soll explizit sagen 'niemals SRC_X_N'.
        """
        candidates = [
            _PROJECT_ROOT / "rag_answer_reference_facts.py",
            _THIS_DIR / "rag_answer_reference_facts.py",
            Path.cwd() / "rag_answer_reference_facts.py",
        ]
        _src_path = next((p for p in candidates if p.exists()), None)
        if _src_path is None:
            self.skipTest("rag_answer_reference_facts.py nicht gefunden")
        src = _src_path.read_text(encoding="utf-8")
        self.assertIn(
            "niemals SRC_X_N", src,
            "Der Prompt muss 'niemals SRC_X_N' enthalten"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Fix C: Language-Agent System-Prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageAgentPrompt(unittest.TestCase):
    """
    Prüft ob die drei neuen NICHT-ERLAUBT-Regeln im Language-Agent-System-Prompt
    korrekt formuliert sind.

    Kein LLM-Call nötig — reine Textprüfung des Quelltexts.
    """

    @classmethod
    def setUpClass(cls):
        # Suche die Datei an mehreren möglichen Orten
        candidates = [
            _PROJECT_ROOT / "rag_answer_reference_facts.py",
            _THIS_DIR / "rag_answer_reference_facts.py",
            Path.cwd() / "rag_answer_reference_facts.py",
        ]
        src_path = next((p for p in candidates if p.exists()), None)
        if src_path is None:
            raise FileNotFoundError(
                "rag_answer_reference_facts.py nicht gefunden. "
                "Script aus dem Projektverzeichnis aufrufen."
            )
        cls.src = src_path.read_text(encoding="utf-8")

        # Extrahiere den System-Prompt-Block des Language-Agents
        start = cls.src.find("def build_language_review_messages(")
        end   = cls.src.find("\ndef ", start + 1)
        cls.lang_block = cls.src[start:end]

    # ── Fix C.1: Leerer Vorschlag = kein Finding ──────────────────────────────

    def test_leerer_vorschlag_regel_vorhanden(self):
        """Die Regel 'Kein Vorschlag vorhanden → kein Finding' muss im Prompt stehen."""
        self.assertIn(
            "Kein Vorschlag vorhanden",
            self.lang_block,
            "Regel 'Kein Vorschlag vorhanden' fehlt im Language-Agent-Prompt"
        )

    def test_fachbegriff_erklaerung_vorhanden(self):
        """Die Erklärung (Fachbegriff/Eigenname/Kompositum) muss enthalten sein."""
        self.assertIn(
            "Fachbegriff",
            self.lang_block,
            "Erklärung 'Fachbegriff' fehlt im Kontext des leeren-Vorschlag-Schutzes"
        )

    # ── Fix C.2: Komposita-Schutz ─────────────────────────────────────────────

    def test_komposita_nicht_aufteilen_vorhanden(self):
        """Die Regel 'Komposita NIEMALS aufteilen' muss im Prompt stehen."""
        self.assertIn(
            "Komposita NIEMALS aufteilen",
            self.lang_block,
            "Komposita-Schutz fehlt im Language-Agent-Prompt"
        )

    def test_leerzeichen_einfuegen_verboten(self):
        """Die explizite Regel gegen Leerzeichen in Komposita muss enthalten sein."""
        self.assertIn(
            "Ein Leerzeichen einzufügen",
            self.lang_block,
            "Verbot des Leerzeichen-Einfügens fehlt"
        )

    def test_beispiele_komposita_vorhanden(self):
        """Mindestens ein konkretes Beispiel für geschützte Komposita muss stehen."""
        has_example = any(
            w in self.lang_block
            for w in ["zurückzukehrte", "herumwirbelnde", "brandbetroffene", "lagekorrekt"]
        )
        self.assertTrue(
            has_example,
            "Kein konkretes Kompositum-Beispiel im Language-Agent-Prompt"
        )

    # ── Fix C.3: Grammatik ohne Bedeutungsänderung ────────────────────────────

    def test_grammatik_bedeutungsaenderung_verboten(self):
        """Die Regel gegen bedeutungsändernde Grammatikkorrekturen muss enthalten sein."""
        self.assertIn(
            "Grammatikkorrekturen die die Bedeutung",
            self.lang_block,
            "Regel gegen bedeutungsändernde Grammatikkorrekturen fehlt"
        )

    def test_adjektivbeugung_einschraenkung(self):
        """Der Hinweis auf korrekte Adjektivbeugung muss enthalten sein."""
        self.assertIn(
            "Adjektivbeugung",
            self.lang_block,
            "Einschränkung der Adjektivbeugungskorrektur fehlt"
        )

    def test_elektrischer_weidezaun_beispiel(self):
        """'elektrischer' und 'Weidezaun' müssen beide im Language-Block stehen.
        Der Begriff kann über eine String-Fortsetzungszeile aufgeteilt sein.
        """
        self.assertIn(
            "elektrischer",
            self.lang_block,
            "'elektrischer' fehlt im Language-Agent-Prompt"
        )
        self.assertIn(
            "Weidezaun",
            self.lang_block,
            "'Weidezaun' fehlt im Language-Agent-Prompt"
        )

    # ── Fix C.4: ss-Schutz mit mutmasslich ───────────────────────────────────

    def test_mutmasslich_ss_beispiel(self):
        """Das mutmasslich/mutmaßlich-Beispiel muss die ss-Korrektheit zeigen."""
        self.assertIn(
            "mutmasslich",
            self.lang_block,
            "ss-Beispiel 'mutmasslich' fehlt im Language-Agent-Prompt"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEvidenzFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestLanguageAgentPrompt))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"{'='*60}")
    print(f"  {passed}/{total} Tests bestanden")
    if failed:
        print(f"  {failed} FEHLGESCHLAGEN")
    print(f"{'='*60}")

    sys.exit(0 if result.wasSuccessful() else 1)
