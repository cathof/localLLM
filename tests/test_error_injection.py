"""
Test: Synthetische Fehlerinjektion via qwen2.5:72b-instruct-q4_K_M (Ollama)
Fehlerklasse: STRUKT_BEFUND_BESCHREIBUNG
Aufruf: python test_error_injection.py
"""

import json
import requests

OLLAMA_URL = "http://10.0.0.2:11434/api/chat"
MODEL      = "qwen2.5:72b-instruct-q4_K_M"

# ── Testsätze ────────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    {
        "id": "s01",
        "text": (
            "Die Blutalkoholkonzentration der beschuldigten Person wurde "
            "um 01:43 Uhr mit 1.42 Promille gemessen."
        ),
        "comment": "Messwert + Uhrzeit → gute Kandidaten für Substitution"
    },
    {
        "id": "s02",
        "text": (
            "Der Auftrag zur Erstellung des forensischen Gutachtens wurde "
            "von der Staatsanwaltschaft Winterthur/Unterland am 14. März 2024 erteilt."
        ),
        "comment": "Institution + Datum → klassische Entitäts-Substitution"
    },
    {
        "id": "s03",
        "text": (
            "Die Spuren wiesen eine Länge von 23.4 Metern auf und verliefen "
            "parallel zur Fahrbahnmarkierung in einem Winkel von etwa 8 Grad."
        ),
        "comment": "Zwei Messwerte + räumliche Beschreibung → mehrere Kandidaten"
    },
]

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Du bist ein Assistent, der synthetische Fehler in deutsche forensische Gutachtentexte injiziert.
Deine Aufgabe ist es, in einem gegebenen Satz einen einzelnen, minimal-invasiven Fehler der Klasse STRUKT_BEFUND_BESCHREIBUNG einzubauen.

## Definition STRUKT_BEFUND_BESCHREIBUNG
Ein Sachverhalt wird falsch oder unpräzise beschrieben. Typisch: falscher Messwert, falsche Institution, falsche Person, falsche Uhrzeit, falscher Ort – aber immer so, dass der fehlerhafte Wert im Kontext eines echten Gutachtens plausibel wirkt.

## Regeln
1. Ändere AUSSCHLIESSLICH einen einzigen Span im Satz. Kein Zeichen ausserhalb dieses Spans darf verändert werden.
2. Der injizierte Wert muss realistisch klingen (z.B. ähnliche Zahl, ähnlicher Institutionsname aus der Schweiz).
3. Der injizierte Wert muss klar falsch sein, wenn man die Quelle kennt.
4. Erfinde keine Entitäten, die in der Schweizer Rechtspraxis nicht existieren.
5. Antworte AUSSCHLIESSLICH mit einem JSON-Objekt. Kein erklärender Text davor oder danach.

## Ausgabeformat (exakt einhalten)
{
  "original_span": "<der originale, zu ersetzende Textspan>",
  "injected_span": "<der falsche Ersatzwert>",
  "modified_sentence": "<der vollständige Satz mit dem injizierten Fehler>",
  "subclass_id": "STRUKT_BEFUND_BESCHREIBUNG",
  "change_type_id": "CHANGE_FACHLICH",
  "severity_id": "HIGH",
  "rationale": "<ein Satz, warum dieser Fehler inhaltlich relevant ist>"
}
"""

# ── User Prompt Template ──────────────────────────────────────────────────────

def build_user_prompt(sentence: str) -> str:
    return f"""\
Injiziere einen Fehler der Klasse STRUKT_BEFUND_BESCHREIBUNG in den folgenden Satz.

Satz:
\"\"\"{sentence}\"\"\"
"""

# ── Ollama Call ───────────────────────────────────────────────────────────────

def call_qwen(sentence: str) -> dict:
    payload = {
        "model": MODEL,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 512,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(sentence)},
        ],
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

# ── Validation ────────────────────────────────────────────────────────────────

def validate(result: dict, original: str) -> list[str]:
    issues = []
    required = ["original_span", "injected_span", "modified_sentence",
                "subclass_id", "change_type_id", "severity_id", "rationale"]
    for key in required:
        if key not in result:
            issues.append(f"Fehlendes Feld: {key}")

    if "original_span" in result and result["original_span"] not in original:
        issues.append(f"original_span nicht im Originalsatz gefunden: '{result['original_span']}'")

    if "modified_sentence" in result and "injected_span" in result:
        if result["injected_span"] not in result["modified_sentence"]:
            issues.append("injected_span nicht in modified_sentence enthalten")

    if "modified_sentence" in result and "original_span" in result:
        if result["original_span"] in result["modified_sentence"]:
            issues.append("original_span noch in modified_sentence vorhanden (Ersetzung fehlgeschlagen)")

    if result.get("subclass_id") != "STRUKT_BEFUND_BESCHREIBUNG":
        issues.append(f"Falsche subclass_id: {result.get('subclass_id')}")

    return issues

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Modell : {MODEL}")
    print(f"Fehlerklasse: STRUKT_BEFUND_BESCHREIBUNG")
    print("=" * 70)

    all_results = []
    out_path = "injection_test_results.json"

    try:
        for entry in TEST_SENTENCES:
            sid      = entry["id"]
            sentence = entry["text"]
            comment  = entry["comment"]
            content  = ""

            print(f"\n[{sid}] {comment}")
            print(f"  Original : {sentence}")

            try:
                raw     = call_qwen(sentence)
                content = raw["message"]["content"].strip()

                # JSON aus der Antwort extrahieren (Modell liefert manchmal ```json ... ```)
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()

                result = json.loads(content)

            except requests.RequestException as e:
                print(f"  [FEHLER] Ollama nicht erreichbar: {e}")
                all_results.append({"sentence_id": sid, "error": str(e)})
                continue
            except json.JSONDecodeError as e:
                print(f"  [FEHLER] JSON-Parsing fehlgeschlagen: {e}")
                print(f"  Rohantwort: {content[:400]}")
                all_results.append({"sentence_id": sid, "error": f"JSON: {e}", "raw": content[:400]})
                continue
            except Exception as e:
                print(f"  [FEHLER] Unerwarteter Fehler: {e}")
                all_results.append({"sentence_id": sid, "error": str(e)})
                continue

            issues = validate(result, sentence)

            print(f"  Injiziert: {result.get('modified_sentence', '–')}")
            print(f"  Span alt : '{result.get('original_span', '–')}'")
            print(f"  Span neu : '{result.get('injected_span', '–')}'")
            print(f"  Rationale: {result.get('rationale', '–')}")

            if issues:
                print(f"  [VALIDATION] {len(issues)} Problem(e):")
                for iss in issues:
                    print(f"    ✗ {iss}")
            else:
                print(f"  [VALIDATION] OK")

            all_results.append({
                "sentence_id":       sid,
                "original":          sentence,
                "result":            result,
                "validation_ok":     len(issues) == 0,
                "validation_issues": issues,
            })

    finally:
        # Immer schreiben — auch bei halbfertigem Lauf
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        ok  = sum(1 for r in all_results if r.get("validation_ok", False))
        tot = len(all_results)
        print("\n" + "=" * 70)
        print(f"Ergebnis: {ok}/{tot} Injektionen valide")
        print(f"Resultate gespeichert: {out_path}")


if __name__ == "__main__":
    main()
