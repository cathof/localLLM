#!/usr/bin/env python3
"""Streamlit GUI — RAG question answering and case ingestion."""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

import streamlit as st

PROJECT_DIR = Path(__file__).parent
CASES_DIR = PROJECT_DIR / "cases"

st.set_page_config(
    page_title="RAG Anfrage",
    page_icon="🔍",
    layout="wide",
)

st.title("Lokales LLM")

tab_qa, tab_ingest, tab_detect = st.tabs(["Frage stellen", "Neuen Fall hinzufügen", "Fehlererkennung"])


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(cmd: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
        timeout=timeout,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Q&A
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sources_json(text: str) -> Optional[dict]:
    """Find the first valid JSON object that contains a 'sources' key."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict) and "sources" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
                start = -1
    return None


_SEP_EQ = "=" * 90
_SEP_DASH = "-" * 90


def _extract_section(text: str, header: str) -> str:
    pattern = (
            re.escape(_SEP_EQ)
            + r"\n"
            + re.escape(header)
            + r"\n"
            + re.escape(_SEP_EQ)
            + r"\n([\s\S]*?)(?=\n[=\-]{80,}|\Z)"
    )
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _extract_quellen(text: str) -> str:
    pattern = (
            re.escape(_SEP_DASH)
            + r"\nQUELLEN\n"
            + re.escape(_SEP_DASH)
            + r"\n([\s\S]*)$"
    )
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _run_query(q: str) -> dict:
    proc = _run([
        sys.executable,
        str(PROJECT_DIR / "rag_answer_reference_facts.py"),
        "--question", q,
        "--print_sources",
        "--print_context",
    ])
    out = proc.stdout
    return {
        "sources": _extract_sources_json(out),
        "answer": _extract_section(out, "ANSWER"),
        "context": _extract_section(out, "RETRIEVED CONTEXT"),
        "quellen": _extract_quellen(out),
        "logs": [l for l in out.splitlines() if l.startswith(("[INFO]", "[WARN]"))],
        "stdout_raw": out,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


with tab_qa:
    question = st.text_area(
        "Frage eingeben",
        placeholder="z. B. Dürfen Personendaten im Ausland bekanntgegeben werden?",
        height=100,
    )
    run_btn = st.button("Frage stellen", type="primary", disabled=not question.strip())

    if run_btn and question.strip():
        with st.spinner("Retrieval und LLM-Antwort laufen …"):
            try:
                result = _run_query(question.strip())
            except subprocess.TimeoutExpired:
                st.error("Zeitüberschreitung (> 10 min). Ist Ollama erreichbar?")
                st.stop()

        if result["returncode"] != 0 and not result["answer"]:
            st.error("Das Skript endete mit einem Fehler.")
            with st.expander("Fehlerausgabe"):
                st.code(result["stderr"] or result["stdout_raw"])
            st.stop()

        # Answer
        st.subheader("Antwort")
        if result["answer"]:
            st.markdown(result["answer"])
        else:
            st.warning("Keine Antwort erhalten.")

        st.divider()

        # Sources table
        sources_data = result["sources"]
        if sources_data and sources_data.get("sources"):
            src_list = sources_data["sources"]
            with st.expander(f"Quellen — {len(src_list)} Treffer", expanded=True):
                if sources_data.get("multi_queries"):
                    st.caption("Suchanfragen: " + "  ·  ".join(sources_data["multi_queries"]))
                rows = [
                    {
                        "#": s.get("n", i + 1),
                        "Dokument": s.get("source_name") or "—",
                        "Score": round(float(s.get("score", 0)), 4),
                        "Typ": s.get("source_kind") or "—",
                        "Chunk": s["chunk_index"] if s.get("chunk_index") is not None else "—",
                        "Suchanfrage": (s.get("retrieval_query") or "")[:70],
                    }
                    for i, s in enumerate(src_list)
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # Context chunks
        if result["context"]:
            with st.expander("Abgerufener Kontext"):
                for chunk in re.split(r"\n\n(?=\[)", result["context"]):
                    lines = chunk.strip().splitlines()
                    if not lines:
                        continue
                    st.markdown(f"**{lines[0]}**")
                    body = "\n".join(lines[1:]).strip()
                    if body:
                        st.text(body)
                    st.divider()

        # Logs
        if result["logs"]:
            with st.expander("Logs"):
                st.text("\n".join(result["logs"]))


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Ingest new case
# ══════════════════════════════════════════════════════════════════════════════

def _existing_cases() -> list[str]:
    if not CASES_DIR.exists():
        return []
    return sorted(
        d.name for d in CASES_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _case_files(case_id: str) -> dict[str, list[str]]:
    """Return {subfolder: [filenames]} for a given case_id."""
    case_dir = CASES_DIR / case_id
    if not case_dir.exists():
        return {}
    tree: dict[str, list[str]] = {}
    for subdir in sorted(case_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            filenames = sorted(
                fp.name for fp in subdir.iterdir()
                if fp.is_file() and not fp.name.startswith(".")
            )
            tree[subdir.name] = filenames
    return tree


DATA_DIR = PROJECT_DIR / "data"
DETECTION_DIR = PROJECT_DIR / "detection"


def _existing_rules_docs() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        f.name for f in DATA_DIR.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )


def _run_append_case(case_id: str) -> dict:
    proc = _run([
        sys.executable,
        str(PROJECT_DIR / "rag_append_case.py"),
        "--case_id", case_id,
    ], timeout=1200)
    return {"logs": proc.stdout + proc.stderr, "returncode": proc.returncode}


def _run_rebuild_rules() -> dict:
    """Full re-ingest + re-embed of the rules store (data/ → artefacts/prepared_rules.jsonl → artefacts/embeddings_rules.npz)."""
    step1 = _run([
        sys.executable,
        str(PROJECT_DIR / "importDocuments_structural.py"),
    ], timeout=1800)
    if step1.returncode != 0:
        return {
            "logs": step1.stdout + step1.stderr,
            "returncode": step1.returncode,
            "failed_at": "Schritt 1 (Chunking)",
        }

    step2 = _run([
        sys.executable,
        str(PROJECT_DIR / "embed_e5.py"),
    ], timeout=1800)
    return {
        "logs": step1.stdout + step1.stderr + "\n" + step2.stdout + step2.stderr,
        "returncode": step2.returncode,
        "failed_at": "Schritt 2 (Einbetten)" if step2.returncode != 0 else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Fehlererkennung helpers
# ══════════════════════════════════════════════════════════════════════════════

_CASE_DOCS_DIR = PROJECT_DIR / "case_documents"


def _existing_case_documents() -> list[str]:
    if not _CASE_DOCS_DIR.exists():
        return []
    return sorted(
        f.name for f in _CASE_DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".docx" and not f.name.startswith(".")
    )


def _existing_findings_files() -> list[str]:
    if not DETECTION_DIR.exists():
        return []
    return sorted(
        (f.name for f in DETECTION_DIR.iterdir()
         if f.is_file() and f.suffix == ".json" and not f.name.startswith(".")),
        reverse=True,
    )


def _parse_saved_path(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Gespeichert"):
            parts = s.split(":", 1)
            return parts[1].strip() if len(parts) > 1 else None
    return None


def _run_detect(doc_path: str) -> dict:
    proc = _run([
        sys.executable,
        str(PROJECT_DIR / "llm_error_detector.py"),
        "--mode", "detect",
        "--document", doc_path,
    ], timeout=1800)
    return {
        "logs": proc.stdout + proc.stderr,
        "returncode": proc.returncode,
        "findings_path": _parse_saved_path(proc.stdout),
    }


def _run_apply(doc_path: str, findings_path: str) -> dict:
    proc = _run([
        sys.executable,
        str(PROJECT_DIR / "llm_error_detector.py"),
        "--mode", "apply",
        "--document", doc_path,
        "--findings", findings_path,
    ], timeout=300)
    return {
        "logs": proc.stdout + proc.stderr,
        "returncode": proc.returncode,
        "corrected_path": _parse_saved_path(proc.stdout),
    }


def _run_to_gt(doc_path: str, findings_path: str) -> dict:
    proc = _run([
        sys.executable,
        str(PROJECT_DIR / "llm_error_detector.py"),
        "--mode", "to_gt",
        "--document", doc_path,
        "--findings", findings_path,
    ], timeout=120)
    return {
        "logs": proc.stdout + proc.stderr,
        "returncode": proc.returncode,
        "gt_path": _parse_saved_path(proc.stdout),
    }


# ── Tab 2 layout ───────────────────────────────────────────────────────────────

with tab_ingest:
    st.subheader("Material in den RAG-Store aufnehmen")

    store_choice = st.radio(
        "Welchen Store befüllen?",
        ["Fallmaterial  (Anhängen, kein Überschreiben)", "Regelwerk  (Vollständiger Neuaufbau)"],
        horizontal=True,
    )

    st.divider()

    # ── Constants ─────────────────────────────────────────────────────────────
    _CASE_SUBFOLDERS = ["auftrag", "berichte", "email", "photo", "sonstiges"]
    _SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "pptx"]
    CASE_DOCUMENTS_DIR = PROJECT_DIR / "case_documents"

    if store_choice.startswith("Fallmaterial"):

        upload_target = st.radio(
            "Was möchten Sie hochladen?",
            [
                "Gutachtendokument  →  case_documents/",
                "Zusatzmaterial  →  cases/<case_id>/<unterordner>/",
            ],
            horizontal=True,
        )

        st.divider()

        # ── A1: upload the case document itself (case_documents/) ─────────────
        if upload_target.startswith("Gutachten"):
            st.caption(
                "Das Gutachtendokument wird direkt nach `case_documents/` gespeichert — "
                "kein Unterordner, kein Einbetten erforderlich."
            )
            existing_docs = sorted(
                f.name for f in CASE_DOCUMENTS_DIR.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ) if CASE_DOCUMENTS_DIR.exists() else []

            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown(f"**Vorhandene Dokumente** ({len(existing_docs)})")
                if existing_docs:
                    st.dataframe(
                        [{"Datei": d} for d in existing_docs],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Noch keine Dokumente in `case_documents/`.")

            with col_right:
                doc_files = st.file_uploader(
                    "Gutachtendokument(e) hochladen",
                    type=_SUPPORTED_TYPES,
                    accept_multiple_files=True,
                )
                save_doc_btn = st.button(
                    "Speichern",
                    disabled=not doc_files,
                )

            if save_doc_btn and doc_files:
                CASE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
                saved = []
                for uf in doc_files:
                    (CASE_DOCUMENTS_DIR / uf.name).write_bytes(uf.getbuffer())  # type: ignore[union-attr]
                    saved.append(uf.name)  # type: ignore[union-attr]
                st.success(
                    f"{len(saved)} Datei(en) gespeichert nach `case_documents/`: "
                    + ", ".join(saved)
                )

        # ── A2: upload supporting material (cases/<id>/<subfolder>/) ──────────
        else:
            st.caption(
                "Zusatzmaterial wird nach `cases/<case_id>/<unterordner>/` gespeichert. "
                "Fehlende Ordner werden angelegt. "
                "Anschliessend können die Dateien mit **In Store einbetten** verarbeitet werden."
            )
            st.caption("Befehlszeile: `python rag_append_case.py --case_id <case_id>`")

            existing_cases = _existing_cases()
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("**Vorhandene Fälle**")
                if existing_cases:
                    st.dataframe(
                        [{"Fall": c} for c in existing_cases],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(f"Kein Fall-Ordner unter `{CASES_DIR}` gefunden.")

            with col_right:
                case_id_input = st.text_input(
                    "Case-ID",
                    placeholder="z. B. case_09",
                    help="Bestehende oder neue Case-ID. Fehlende Ordner werden automatisch erstellt.",
                )
                subfolder_input = st.selectbox(
                    "Unterordner",
                    options=_CASE_SUBFOLDERS,
                    help="Dokumenttyp bestimmt den Unterordner.",
                )
                uploaded_files = st.file_uploader(
                    "Dateien hochladen",
                    type=_SUPPORTED_TYPES,
                    accept_multiple_files=True,
                    help=f"Unterstützte Formate: {', '.join('.' + t for t in _SUPPORTED_TYPES)}",
                )
                save_btn = st.button(
                    "Dateien speichern",
                    disabled=not (case_id_input.strip() and uploaded_files),
                )

            if save_btn and case_id_input.strip() and uploaded_files:
                cid = case_id_input.strip()
                dest_dir = CASES_DIR / cid / subfolder_input
                dest_dir.mkdir(parents=True, exist_ok=True)
                saved, overwritten = [], []
                for uf in uploaded_files:
                    dest = dest_dir / uf.name  # type: ignore[union-attr]
                    if dest.exists():
                        overwritten.append(uf.name)  # type: ignore[union-attr]
                    dest.write_bytes(uf.getbuffer())  # type: ignore[union-attr]
                    saved.append(uf.name)  # type: ignore[union-attr]
                st.success(
                    f"{len(saved)} Datei(en) gespeichert nach `cases/{cid}/{subfolder_input}/`"
                    + (f" ({len(overwritten)} überschrieben)" if overwritten else "")
                )
                file_tree = _case_files(cid)
                if file_tree:
                    st.markdown(f"**Aktueller Inhalt von `cases/{cid}/`**")
                    for subname, names in file_tree.items():
                        with st.expander(f"`{subname}/` — {len(names)} Datei(en)"):
                            for fname in names:
                                st.text(f"  {fname}")

            elif case_id_input.strip() and not uploaded_files:
                file_tree = _case_files(case_id_input.strip())
                if file_tree:
                    st.markdown(f"**Inhalt von `cases/{case_id_input.strip()}/`**")
                    for subname, names in file_tree.items():
                        with st.expander(f"`{subname}/` — {len(names)} Datei(en)"):
                            for fname in names:
                                st.text(f"  {fname}")
                else:
                    st.info(
                        f"Ordner `cases/{case_id_input.strip()}/` existiert noch nicht — "
                        "wird beim Speichern angelegt."
                    )

            st.divider()
            ingest_btn = st.button(
                "In Store einbetten",
                type="primary",
                disabled=not (
                        case_id_input.strip()
                        and (CASES_DIR / case_id_input.strip()).exists()
                ),
                help="Chunking und Einbetten aller neuen Dateien für diese Case-ID.",
            )

            if ingest_btn and case_id_input.strip():
                cid = case_id_input.strip()
                with st.spinner(f"Einbetten und Zusammenführen für {cid} …"):
                    try:
                        res = _run_append_case(cid)
                    except subprocess.TimeoutExpired:
                        st.error("Zeitüberschreitung (> 20 min).")
                        st.stop()

                if res["returncode"] == 0:
                    st.success(f"Fall **{cid}** wurde erfolgreich in den Store aufgenommen.")
                else:
                    st.error(f"Fehler beim Hinzufügen von **{cid}**.")
                with st.expander("Ausgabe / Logs", expanded=res["returncode"] != 0):
                    st.text(res["logs"])

    # ── Option B: rules store full rebuild ────────────────────────────────────
    else:
        st.warning(
            "Der Regelwerk-Store wird vollständig neu aufgebaut. "
            "Alle Dokumente in `data/` werden neu eingelesen und eingebettet. "
            "Das kann mehrere Minuten dauern."
        )
        st.caption(
            "Befehlszeile: `python importDocuments_structural.py` "
            "→ `python embed_e5.py`"
        )

        rules_docs = _existing_rules_docs()
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"**Dokumente in `data/`** ({len(rules_docs)} Dateien)")
            if rules_docs:
                st.dataframe(
                    [{"Datei": d} for d in rules_docs],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"Keine Dokumente in `{DATA_DIR}` gefunden.")

        with col_right:
            uploaded = st.file_uploader(
                "Neues Dokument hochladen (optional)",
                type=["pdf", "docx", "txt"],
                help="Die Datei wird nach `data/` gespeichert, danach den Store neu aufbauen.",
            )
            if uploaded is not None:
                dest = DATA_DIR / uploaded.name
                if dest.exists():
                    st.warning(f"`{uploaded.name}` existiert bereits in `data/` und wird überschrieben.")
                dest.write_bytes(uploaded.getbuffer())
                st.success(f"`{uploaded.name}` gespeichert.")

            rebuild_btn = st.button(
                "Regelwerk neu aufbauen",
                type="primary",
                disabled=not DATA_DIR.exists(),
            )

        if rebuild_btn:
            with st.spinner("Schritt 1/2: Chunking (importDocuments_structural.py) …"):
                try:
                    res = _run_rebuild_rules()
                except subprocess.TimeoutExpired:
                    st.error("Zeitüberschreitung (> 30 min).")
                    st.stop()

            if res["returncode"] == 0:
                st.success("Regelwerk-Store wurde erfolgreich neu aufgebaut.")
            else:
                st.error(f"Fehler in {res.get('failed_at', 'unbekanntem Schritt')}.")
            with st.expander("Ausgabe / Logs", expanded=res["returncode"] != 0):
                st.text(res["logs"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Fehlererkennung
# ══════════════════════════════════════════════════════════════════════════════

_STATUS_OPTIONS = ["pending", "confirmed", "rejected", "corrected"]
_SEVERITIES_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_HUMAN_CORRECTION_OPTIONS = ["corrected", "rejected"]

with tab_detect:
    st.subheader("Fehlererkennung in Gutachten")

    # Session state init
    for _key in ("det_findings", "det_findings_path", "det_doc_path"):
        if _key not in st.session_state:
            st.session_state[_key] = [] if _key == "det_findings" else ""

    # ── Schritt 1: Erkennen ───────────────────────────────────────────────────
    st.markdown("### Schritt 1 — Erkennen")

    doc_options = _existing_case_documents()
    col_s1_l, col_s1_r = st.columns(2)

    with col_s1_l:
        st.markdown("**Dokument auswählen**")
        if doc_options:
            selected_doc = st.selectbox(
                "Gutachtendokument",
                doc_options,
                label_visibility="collapsed",
            )
            derived_case_id = Path(selected_doc).stem
            st.caption(f"Case-ID (abgeleitet): `{derived_case_id}`")
            active_doc_path = str(_CASE_DOCS_DIR / selected_doc)
        else:
            st.info("Keine `.docx`-Dateien in `case_documents/` gefunden.")
            selected_doc = None
            active_doc_path = ""
            derived_case_id = ""

        detect_btn = st.button(
            "Erkennung starten",
            type="primary",
            disabled=not doc_options,
            help="Startet LLM-Agenten zur Fehlererkennung. Kann mehrere Minuten dauern.",
        )

    with col_s1_r:
        st.markdown("**Oder vorhandene Findings laden**")
        existing_findings_files = _existing_findings_files()
        if existing_findings_files:
            sel_findings_file = st.selectbox(
                "Findings-Datei",
                existing_findings_files,
                label_visibility="collapsed",
            )
            load_findings_btn = st.button("Laden")
        else:
            st.info("Noch keine Findings-Dateien in `detection/`.")
            sel_findings_file = None
            load_findings_btn = False

    if detect_btn and doc_options and selected_doc:
        with st.spinner("LLM-Agenten laufen … (kann mehrere Minuten dauern)"):
            try:
                det_res = _run_detect(active_doc_path)
            except subprocess.TimeoutExpired:
                st.error("Zeitüberschreitung (> 30 min). Ist Ollama erreichbar?")
                st.stop()
        if det_res["returncode"] == 0 and det_res["findings_path"]:
            fp = det_res["findings_path"]
            loaded = json.loads(Path(fp).read_text(encoding="utf-8"))
            st.session_state.det_findings = loaded
            st.session_state.det_findings_path = fp
            st.session_state.det_doc_path = active_doc_path
            st.success(f"{len(loaded)} Finding(s) erkannt. Gespeichert: `{fp}`")
        else:
            st.error("Erkennung fehlgeschlagen.")
        with st.expander("Ausgabe / Logs", expanded=det_res["returncode"] != 0):
            st.text(det_res["logs"])

    if load_findings_btn and sel_findings_file:
        fp = str(DETECTION_DIR / sel_findings_file)
        try:
            loaded = json.loads(Path(fp).read_text(encoding="utf-8"))
            st.session_state.det_findings = loaded
            st.session_state.det_findings_path = fp
            if active_doc_path:
                st.session_state.det_doc_path = active_doc_path
            st.success(f"{len(loaded)} Finding(s) aus `{sel_findings_file}` geladen.")
        except Exception as exc:
            st.error(f"Fehler beim Laden: {exc}")

    # ── Schritt 2: Prüfen ─────────────────────────────────────────────────────
    findings: list[dict] = st.session_state.det_findings

    if not findings:
        st.info("Noch keine Findings. Erkennung starten oder vorhandene Datei laden.")
    else:
        st.divider()
        st.markdown(f"### Schritt 2 — Prüfen  ({len(findings)} Findings)")

        # Summary: counts by agent × severity
        _agents = sorted({f.get("agent", "?") for f in findings})
        agent_counts: dict[str, Counter] = {a: Counter() for a in _agents}
        for _f in findings:
            agent_counts[_f.get("agent", "?")][_f.get("severity_id", "?")] += 1

        summary_rows = [
            {
                "Agent": ag,
                **{sev: agent_counts[ag].get(sev, 0) for sev in _SEVERITIES_ORDER},
                "Total": sum(agent_counts[ag].values()),
            }
            for ag in _agents
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        status_counts: Counter = Counter(f.get("status", "pending") for f in findings)
        status_cols = st.columns(len(_STATUS_OPTIONS))
        for _i, _s in enumerate(_STATUS_OPTIONS):
            status_cols[_i].metric(_s, status_counts.get(_s, 0))

        st.divider()

        # Editable table
        edit_rows = [
            {
                "ID": f.get("finding_id", ""),
                "Agent": f.get("agent", ""),
                "Schwere": f.get("severity_id", ""),
                "Klasse": f.get("subclass_id", ""),
                "Original": str(f.get("original_span") or ""),
                "Vorschlag": str(f.get("corrected_span") or "")[:60],
                "Status": (
                    f.get("status")
                    if f.get("status") in _STATUS_OPTIONS
                    else "pending"
                ),
                "Korrektur (manuell)": (
                    f.get("human_correction")
                    if f.get("human_correction") in _HUMAN_CORRECTION_OPTIONS
                    else None
                ),
            }
            for f in findings
        ]
        edited_df = st.data_editor(
            pd.DataFrame(edit_rows),
            use_container_width=True,
            hide_index=True,
            disabled=["ID", "Agent", "Schwere", "Klasse", "Original", "Vorschlag"],
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=_STATUS_OPTIONS,
                    required=True,
                ),
                "Korrektur (manuell)": st.column_config.SelectboxColumn(
                    "Korrektur (manuell)",
                    options=_HUMAN_CORRECTION_OPTIONS,
                    required=False,
                    help="Manuelle Bewertung: confirmed, corrected oder rejected.",
                ),
            },
            key="findings_editor",
        )

        # Detail panel
        st.markdown("**Finding-Details**")
        detail_ids = [r["ID"] for r in edit_rows if r.get("ID")]
        sel_detail = st.selectbox("Finding auswählen", detail_ids, key="detail_select")
        if sel_detail:
            detail = next((f for f in findings if f.get("finding_id") == sel_detail), None)
            if detail:
                with st.expander(f"Detail: {sel_detail}", expanded=True):

                    def _kv(container, label, value):
                        """Zeigt Label links, Wert rechts. Durch das feste
                        Breitenverhältnis [1, 2] beginnen alle Werte innerhalb
                        eines Blocks an derselben senkrechten Linie."""
                        lab_col, val_col = container.columns([1, 2])
                        lab_col.markdown(f"**{label}**")
                        val_col.markdown(str(value))

                    col_d1, col_d2 = st.columns(2)

                    _kv(col_d1, "Agent:", detail.get("agent", "—"))
                    _kv(col_d1, "Schwere:", detail.get("severity_id", "—"))
                    _kv(col_d1, "Klasse:", detail.get("subclass_id", "—"))
                    _kv(col_d1, "Segment:", detail.get("segment_index", "—"))
                    _kv(col_d1, "Status:", detail.get("status", "—"))

                    _kv(col_d2, "Original:", detail.get("original_span", "—"))
                    _kv(col_d2, "Vorschlag:", detail.get("corrected_span", "—"))
                    if detail.get("human_correction"):
                        _kv(col_d2, "Manuelle Korrektur:", detail["human_correction"])
                    _kv(col_d2, "Begründung:", detail.get("rationale", "—"))


        st.divider()

        # Save findings
        save_btn = st.button("Findings speichern", key="save_findings")
        if save_btn:
            edited_records = edited_df.to_dict("records")
            edited_by_id = {r["ID"]: r for r in edited_records}
            updated = []
            for _f in findings:
                fid = _f.get("finding_id", "")
                row = edited_by_id.get(fid)
                if row:
                    _f = {
                        **_f,
                        "status": row["Status"],
                        "human_correction": row["Korrektur (manuell)"],
                    }
                updated.append(_f)
            fp = st.session_state.det_findings_path
            if fp:
                Path(fp).write_text(
                    json.dumps(updated, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.session_state.det_findings = updated
                st.success(f"Gespeichert: `{fp}`")
            else:
                st.error("Kein Findings-Pfad — Erkennung starten oder Datei laden.")

        # ── Schritt 3: Anwenden ───────────────────────────────────────────────
        st.divider()
        st.markdown("### Schritt 3 — Anwenden")

        _confirmed = [
            f for f in st.session_state.det_findings
            if f.get("status") in ("confirmed", "corrected")
        ]
        st.caption(
            f"{len(_confirmed)} Finding(s) mit Status 'confirmed' oder 'corrected' werden angewandt."
        )

        _doc_for_apply = st.session_state.det_doc_path or active_doc_path
        _fp_for_apply = st.session_state.det_findings_path
        _apply_ready = bool(_doc_for_apply and _fp_for_apply and _confirmed)

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown("**Korrekturen ins Dokument schreiben**")
            apply_btn = st.button(
                "Korrekturen anwenden",
                type="primary",
                disabled=not _apply_ready,
                key="apply_btn",
            )
            if apply_btn:
                with st.spinner("Wende Korrekturen an …"):
                    try:
                        res_apply = _run_apply(_doc_for_apply, _fp_for_apply)
                    except subprocess.TimeoutExpired:
                        st.error("Zeitüberschreitung.")
                        st.stop()
                if res_apply["returncode"] == 0:
                    corrected_p = res_apply["corrected_path"] or ""
                    st.success(f"Gespeichert: `{corrected_p}`")
                    if corrected_p and Path(corrected_p).exists():
                        st.download_button(
                            "Korrigiertes Dokument herunterladen",
                            data=Path(corrected_p).read_bytes(),
                            file_name=Path(corrected_p).name,
                            mime=(
                                "application/vnd.openxmlformats-officedocument"
                                ".wordprocessingml.document"
                            ),
                            key="dl_corrected",
                        )
                else:
                    st.error("Fehler beim Anwenden der Korrekturen.")
                with st.expander("Ausgabe / Logs", expanded=res_apply["returncode"] != 0):
                    st.text(res_apply["logs"])

        with col_a2:
            st.markdown("**Ground Truth exportieren**")
            to_gt_btn = st.button(
                "Ground Truth exportieren",
                disabled=not _apply_ready,
                key="to_gt_btn",
            )
            if to_gt_btn:
                with st.spinner("Exportiere Ground Truth …"):
                    try:
                        res_gt = _run_to_gt(_doc_for_apply, _fp_for_apply)
                    except subprocess.TimeoutExpired:
                        st.error("Zeitüberschreitung.")
                        st.stop()
                if res_gt["returncode"] == 0:
                    gt_p = res_gt["gt_path"] or ""
                    st.success(f"Gespeichert: `{gt_p}`")
                else:
                    st.error("Fehler beim Exportieren der Ground Truth.")
                with st.expander("Ausgabe / Logs", expanded=res_gt["returncode"] != 0):
                    st.text(res_gt["logs"])