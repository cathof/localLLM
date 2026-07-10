#!/usr/bin/env python3
"""
Kurztest für Agent 0 (Referenzfakten) in der gepatchten Fassung von
rag_answer_reference_facts.py.

Abgedeckt:
  1. run_reference_facts_agent parst eine gültige LLM-Antwort korrekt.
  2. Bug-Symptom aus dem Log: leere LLM-Antwort ("") darf NICHT crashen,
     sondern ein gültiges, leeres Skelett liefern.
  3. Der eigentliche Fix: OllamaClient sendet standardmäßig "think": false.
  4. Retry-on-empty: bei aktivem Thinking + leerem Content wird einmal
     mit think=false nachgefragt.
  5. Fallback: lehnt Ollama das think-Feld mit 400 ab, wird es verworfen
     und ohne think erneut gesendet.

Laufen lassen:
  pytest test_reference_agent.py            # mit pytest
  python  test_reference_agent.py           # ohne pytest (eigener Runner)

Es wird kein Ollama-Server und kein echtes Modell benötigt — requests.post
wird gemockt.
"""
from __future__ import annotations

import json

import rag_answer_reference_facts_GRUNDLAGE as rag


# ─────────────────────────── Test-Hilfen ────────────────────────────────────

class FakeLLM(rag.LLMClient):
    """Minimaler LLM-Client, der eine feste Antwort zurückgibt und Aufrufe merkt."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, messages, json_mode=False, schema=None):
        self.calls.append({"messages": messages, "json_mode": json_mode, "schema": schema})
        return self.reply


class FakeResponse:
    """Ersetzt das requests.Response-Objekt für die OllamaClient-Tests."""

    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            err = rag.requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


# Agent 0 übergibt ein Schema; der Inhalt ist für die Agentenlogik egal.
SCHEMA = {"type": "object"}

# Dokumenttext OHNE "Person <Vor> <Nach>"-Kopfzeile, damit der deterministische
# Anreicherungsschritt (enrich_reference_facts_from_document) nichts ergänzt und
# das leere Skelett in Test 2 wirklich leer bleibt.
DOC_TEXT = (
    "Gutachten Brandursache\n"
    "Auftraggeber: Staatsanwaltschaft Oberwallis\n"
    "Vorfall: Wald- und Wiesenbrand\n"
    "Ereignisdatum: 26. März 2022\n"
    "Ort: Hohtenn\n"
)

GOOD_REPLY = json.dumps(
    {
        "case_id": "case_01",
        "facts": {
            "auftraggeber": {
                "value": "Staatsanwaltschaft Oberwallis",
                "source_span": "Auftraggeber: Staatsanwaltschaft Oberwallis",
                "confidence": "high",
            },
            "ereignisdatum": {
                "value": "2022-03-26",
                "source_span": "Ereignisdatum: 26. März 2022",
                "confidence": "high",
            },
            "ort": {"value": "Hohtenn", "source_span": "Ort: Hohtenn", "confidence": "high"},
            "personen": [
                {
                    "name": "Max Muster",
                    "rolle": "Sachverständiger",
                    "source_span": "Sachverständiger: Max Muster",
                    "confidence": "high",
                }
            ],
            "referenz_entitaeten": [],
        },
    },
    ensure_ascii=False,
)


def _patch_post(fake_post):
    """requests.post im Modul temporär ersetzen; gibt das Original zurück."""
    original = rag.requests.post
    rag.requests.post = fake_post
    return original


# ─────────────────────── Tests: run_reference_facts_agent ────────────────────

def test_agent_parses_good_reply():
    llm = FakeLLM(GOOD_REPLY)
    out = rag.run_reference_facts_agent(
        llm, DOC_TEXT, case_id="case_01", schema=SCHEMA, max_chars=4000
    )
    facts = out["facts"]

    assert out["case_id"] == "case_01"
    assert facts["auftraggeber"]["value"] == "Staatsanwaltschaft Oberwallis"
    assert facts["auftraggeber"]["confidence"] == "high"
    assert facts["ereignisdatum"]["value"] == "2022-03-26"
    assert facts["ort"]["value"] == "Hohtenn"

    # Alle Schema-Felder sind vorhanden und normalisiert.
    for key in rag.REFERENCE_FACT_KEYS:
        assert set(facts[key].keys()) == {"value", "source_span", "confidence"}

    assert isinstance(facts["personen"], list) and facts["personen"]
    assert facts["personen"][0]["name"] == "Max Muster"

    # Der Agent ruft das LLM im json_mode mit dem Schema auf.
    assert llm.calls[0]["json_mode"] is True
    assert llm.calls[0]["schema"] is SCHEMA


def test_agent_empty_reply_degrades_gracefully():
    """Bug-Symptom aus dem Log: 0-Zeichen-Antwort → leeres, gültiges Skelett."""
    llm = FakeLLM("")
    out = rag.run_reference_facts_agent(
        llm, DOC_TEXT, case_id="case_01", schema=SCHEMA, max_chars=4000
    )
    facts = out["facts"]

    assert out["case_id"] == "case_01"
    for key in rag.REFERENCE_FACT_KEYS:
        assert facts[key] == {"value": "", "source_span": "", "confidence": "low"}
    assert facts["personen"] == []
    assert facts["referenz_entitaeten"] == []


# ─────────────────────────── Tests: OllamaClient ─────────────────────────────

def test_ollama_disables_thinking_by_default():
    """Der eigentliche Fix: think=false wird mitgesendet."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = dict(json)
        return FakeResponse({"message": {"content": '{"ok": true}'}})

    original = _patch_post(fake_post)
    try:
        client = rag.OllamaClient("http://x", "qwen3.6:35b", options={}, timeout_s=5)
        out = client.chat([{"role": "user", "content": "hi"}], json_mode=True)
    finally:
        rag.requests.post = original

    assert out == '{"ok": true}'
    assert captured["payload"]["think"] is False
    assert captured["payload"]["format"] == "json"


def test_ollama_retries_on_empty_content_when_thinking_enabled():
    """Thinking aktiv + leerer Content → genau ein Retry mit think=false."""
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("think") is False:
            return FakeResponse({"message": {"content": '{"errors":[]}'}})
        # Erster Versuch (Thinking an): leerer Content nach strip().
        return FakeResponse({"message": {"content": "   "}})

    original = _patch_post(fake_post)
    try:
        client = rag.OllamaClient(
            "http://x", "qwen3.6:35b", options={}, timeout_s=5, disable_think=False
        )
        out = client.chat([{"role": "user", "content": "hi"}])
    finally:
        rag.requests.post = original

    assert out == '{"errors":[]}'
    assert len(calls) == 2
    assert "think" not in calls[0]          # erster Call ließ Thinking an
    assert calls[1]["think"] is False        # Retry erzwingt think=false


def test_ollama_drops_think_when_unsupported():
    """Lehnt Ollama das think-Feld mit 400 ab, wird es verworfen und neu gesendet."""
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        calls.append(dict(json))
        if json.get("think") is False:
            return FakeResponse({"error": '"think" is not supported by this model'}, status=400)
        return FakeResponse({"message": {"content": '{"ok":1}'}})

    original = _patch_post(fake_post)
    try:
        client = rag.OllamaClient(
            "http://x", "llama3", options={}, timeout_s=5, disable_think=True
        )
        out = client.chat([{"role": "user", "content": "hi"}], json_mode=True)
    finally:
        rag.requests.post = original

    assert out == '{"ok":1}'
    assert len(calls) == 2
    assert calls[0]["think"] is False        # erster Versuch mit think=false
    assert "think" not in calls[1]           # Fallback ohne think


# ─────────────────────────── Runner ohne pytest ─────────────────────────────

if __name__ == "__main__":
    import sys

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
