# Forensische Dokumentenprüfung mit lokalen LLMs

Ein RAG-gestütztes System zur automatisierten Qualitätsprüfung forensischer Gutachten (z. B. Brandursachenermittlung). Es kombiniert dokumentenstrukturiertes Chunking, dual-store Retrieval (Regelwerke + Fallmaterial) und eine mehrstufige LLM-Agenten-Pipeline, um Gutachten gegen Referenzfakten und fachliche Regeln zu prüfen — vollständig lokal über [Ollama](https://ollama.com), ohne Cloud-LLM-Zugriff.

Entstanden im Rahmen einer Masterarbeit an der ZHAW.

## Warum

Forensische Gutachten müssen formal wie fachlich korrekt sein — falsche Messwerte, verrutschte Hypothesenbewertungen oder unpräzise Formulierungen können Konsequenzen haben. Manuelles Gegenlesen gegen Normen und Fallakten ist aufwändig und fehleranfällig. Dieses Projekt untersucht, ob ein lokal betriebenes LLM-System (aus Datenschutzgründen ohne Cloud-Anbindung) solche Gutachten zuverlässig gegen Referenzwissen prüfen und Fehler vorschlagen kann.

## Wie es funktioniert

1. **Ingestion** — PDFs, DOCX, PPTX und Textdokumente (Regelwerke, Normen, Fallakten) werden eingelesen, normalisiert, strukturell gesplittet und semantisch gechunkt.
2. **Embedding** — Chunks werden mit `intfloat/multilingual-e5-large` vektorisiert und in zwei getrennten Stores abgelegt: **Rules** (Regeln/Normen) und **Materials** (Fallmaterial).
3. **Retrieval-Augmented Generation** — Für ein Gutachten oder eine Freitextfrage holt eine 3-Agenten-Pipeline relevante Evidenz aus beiden Stores:
   - **Agent 1** – Retrieval/Evidenzsuche
   - **Agent 2** – fachlich-inhaltliche Prüfung
   - **Agent 3** – sprachliche Prüfung
4. **Fehlererkennung** — zwei komplementäre Werkzeuge:
   - erkennt **echte** Fehler in einem Originalgutachten (Vorschlag → menschliche Prüfung → Übernahme)
   - erzeugt **synthetische** Fehler, um die Erkennungsgüte des Systems messbar zu machen
5. **Evaluation** — Predictions werden gegen kuratierte Ground Truth verglichen (Precision/Recall je Fehlerklasse, taxonomiebasiert).
6. **GUI** — ein Streamlit-Frontend bündelt Fragen stellen, neue Fälle hinzufügen und Fehlererkennung in einer Oberfläche.

## Architektur

```
Dokumente (PDF/DOCX/PPTX)
        │
        ▼
   Ingestion & Chunking ──▶ strukturierte Chunks
        │
        ▼
   Embedding (multilingual-e5-large) ──▶ zwei Vektor-Stores (Rules / Materials)
        │
        ▼
   3-Agenten RAG-Pipeline (Frage beantworten / Gutachten prüfen)
        │
        ├── echte Fehler im Gutachten erkennen
        └── synthetische Fehler für die Evaluation injizieren
                │
                ▼
        Evaluation gegen Ground Truth (Precision/Recall)

   Streamlit-GUI orchestriert die gesamte Pipeline
```

## Eigenschaften

- **Vollständig lokal** — LLM-Inferenz über Ollama, kein Datenabfluss an externe Cloud-Dienste; relevant für den Umgang mit sensiblen forensischen Fallakten.
- **Dual-Store-Retrieval** — Regelwerke/Normen und Fallmaterial werden getrennt indexiert und gezielt kombiniert abgefragt, statt alles in einen Topf zu werfen.
- **Taxonomiebasierte Fehlerklassen** — Fehler werden nicht nur gefunden, sondern nach einer definierten Taxonomie klassifiziert (z. B. Befundbeschreibung, Redaktion, Absicherung/Hedging, Arithmetik, Hypotheseninkonsistenz).
- **Mensch im Loop** — automatisch erkannte bzw. injizierte Fehler durchlaufen einen expliziten Review-Schritt, bevor sie übernommen oder in die Ground Truth aufgenommen werden.
- **Messbare Qualität** — eine eigene Evaluationspipeline vergleicht Modell-Predictions systematisch gegen Ground-Truth-Daten.

## Projektstruktur

| Bereich | Zweck |
|---|---|
| Ingestion & Embedding | Dokumente einlesen, chunken, vektorisieren |
| RAG-Pipeline | Fragen beantworten, Gutachten mit 3 Agenten prüfen |
| Fehlererkennung | echte Fehler finden, synthetische Fehler für Tests generieren |
| Evaluation | Predictions gegen Ground Truth auswerten |
| GUI | Streamlit-Oberfläche für alle Workflows |
| Pipeline-Skripte | End-to-End-Evaluationslauf über mehrere Testfälle |

Verzeichnisse mit forensischen Falldokumenten (Regelwerke, Fallakten, Gutachten, Ground Truth) sind nicht Teil dieses Repositories — nur die Codebasis und die Ordnerstruktur werden versioniert.

## Status

Forschungsprojekt im Rahmen einer Masterarbeit, aktiv in Entwicklung. Betrieb und Deployment sind an eine spezifische lokale Infrastruktur gebunden und nicht Gegenstand dieser Beschreibung.
