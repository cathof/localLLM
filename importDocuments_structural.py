#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Optional dependencies ─────────────────────────────────────────────────────
#
# PDF text extraction:  pdfplumber (primary)  +  PyMuPDF fallback
# PDF image extraction: PyMuPDF only (pdfplumber has no image API)
# OCR:                  Tesseract via pytesseract (lazy, only for scanned pages)
#
# Removed: pdf2image, pypdf  – both superseded by the libraries above

try:
    import pdfplumber  # pip install pdfplumber
except Exception:
    pdfplumber = None  # type: ignore

try:
    import fitz  # pip install PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    import pytesseract  # pip install pytesseract
    from pytesseract import image_to_string
except Exception:
    pytesseract     = None  # type: ignore
    image_to_string = None  # type: ignore

try:
    from docx import Document  # pip install python-docx
    from docx.document import Document as _Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
except Exception:
    Document  = None  # type: ignore
    _Document = None  # type: ignore
    CT_Tbl    = None  # type: ignore
    CT_P      = None  # type: ignore
    Table     = None  # type: ignore
    _Cell     = None  # type: ignore
    Paragraph = None  # type: ignore

try:
    import tiktoken  # pip install tiktoken
except Exception:
    tiktoken = None  # type: ignore

try:
    from transformers import AutoTokenizer  # pip install transformers
except Exception:
    AutoTokenizer = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import numpy as np  # pip install numpy
except Exception:
    np = None  # type: ignore


SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".pptx"}


# ── .env loader ───────────────────────────────────────────────────────────────

def load_dotenv(dotenv_path: str | Path = ".env") -> None:
    p = Path(dotenv_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() != "" else default

def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return int(v)

def env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return float(v)

def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


load_dotenv(".env")


# ── Defaults ──────────────────────────────────────────────────────────────────

DATA_DIR_DEFAULT        = env_str("DATA_DIR",        "./data")
OUT_JSONL_DEFAULT       = env_str("OUT_JSONL",       "prepared.jsonl")
IMAGE_CACHE_DIR_DEFAULT = env_str("IMAGE_CACHE_DIR", "./image_cache")

CHUNK_SIZE_TOKENS_DEFAULT    = env_int("CHUNK_SIZE_TOKENS",    320)
CHUNK_OVERLAP_TOKENS_DEFAULT = env_int("CHUNK_OVERLAP_TOKENS",  48)
MIN_CHUNK_TOKENS_DEFAULT     = env_int("MIN_CHUNK_TOKENS",       80)

TOKENIZER_BACKEND_DEFAULT = env_str("TOKENIZER_BACKEND", "auto")
TOKENIZER_MODEL_DEFAULT   = env_str("TOKENIZER_MODEL",   "cl100k_base")

ENABLE_SECTION_AWARENESS_DEFAULT = env_bool("ENABLE_SECTION_AWARENESS", True)
ENABLE_SEMANTIC_CHUNKING_DEFAULT = env_bool("ENABLE_SEMANTIC_CHUNKING", True)
SEMANTIC_MODEL_DEFAULT = env_str(
    "SEMANTIC_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# 0.35 (down from 0.42): less aggressive splitting in coherent technical prose
SEMANTIC_THRESHOLD_DEFAULT  = env_float("SEMANTIC_THRESHOLD", 0.35)
SEMANTIC_MIN_UNITS_DEFAULT  = env_int("SEMANTIC_MIN_UNITS", 2)

ENABLE_OCR_DEFAULT             = env_bool("ENABLE_OCR",               False)
ENABLE_OCR_PPTX_DEFAULT        = env_bool("ENABLE_OCR_PPTX",          False)
OCR_LANG_DEFAULT               = env_str ("OCR_LANG",                 "deu")
OCR_MIN_CHARS_PER_PAGE_DEFAULT = env_int ("OCR_MIN_CHARS_PER_PAGE",     80)
OCR_DPI_DEFAULT                = env_int ("OCR_DPI",                   300)

# PyMuPDF image-extraction thresholds
FULL_PAGE_IMAGE_RATIO_DEFAULT = env_float("FULL_PAGE_IMAGE_RATIO", 0.85)
MIN_IMAGE_PX_DEFAULT          = env_int  ("MIN_IMAGE_PX",           100)

TESSERACT_CMD_DEFAULT = os.environ.get("TESSERACT_CMD")
TESSDATA_DIR_DEFAULT  = os.environ.get("TESSDATA_DIR")
SOFFICE_CMD_DEFAULT   = os.environ.get("SOFFICE_CMD")

# ── Vision constants – INFERENCE TIME only, not used during ingestion ─────────
#
# ollama_caption_png() is kept here as a shared utility so the query /
# error-detection pipeline can import it without a separate module.
# It is never called inside the ingestion path.

OLLAMA_BASE_URL_DEFAULT  = env_str("OLLAMA_BASE_URL",  "http://localhost:11434")
VISION_BACKEND_DEFAULT   = env_str("VISION_BACKEND",   "ollama")
VISION_MODEL_DEFAULT     = env_str("VISION_MODEL",     "qwen2.5vl:7b")
VISION_TIMEOUT_S_DEFAULT = env_int("VISION_TIMEOUT_S", 180)

# Prompt for PPTX slides (inference time)
VISION_PROMPT_SLIDE_DEFAULT = env_str(
    "VISION_PROMPT_SLIDE",
    (
        "Beschreibe die Folie präzise:\n"
        "- Titel (falls erkennbar)\n"
        "- Kernaussagen als Bulletpoints (max. 8)\n"
        "- sichtbare Labels/Legenden/Tabellen kurz (falls vorhanden)\n"
        "- falls nur Foto ohne Text: Motiv und Kontext knapp\n"
        "Keine Halluzinationen, nur was sichtbar ist."
    ),
)

# Prompt for PDF figures / charts (inference time)
VISION_PROMPT_FIGURE_DEFAULT = env_str(
    "VISION_PROMPT_FIGURE",
    (
        "Beschreibe dieses Bild aus einem technischen Dokument präzise:\n"
        "- Art des Bildes (Diagramm, Grafik, Foto, Tabelle als Bild, technische Zeichnung)\n"
        "- Kernaussagen und sichtbare Werte\n"
        "- Achsenbeschriftungen, Legenden, Einheiten (falls vorhanden)\n"
        "- technische Maße oder Toleranzangaben (falls vorhanden)\n"
        "Keine Halluzinationen, nur was sichtbar ist."
    ),
)

LOG_LEVEL_DEFAULT   = env_str("LOG_LEVEL",  "INFO").upper()
LOG_EVERY_S_DEFAULT = env_int("LOG_EVERY_S", 10)


# ── Logging ───────────────────────────────────────────────────────────────────

def _lvl(x: str) -> int:
    return {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}.get(x, 20)


def log(
        msg: str,
        *,
        level: str = "INFO",
        cfg: Optional[dict] = None,
        err: bool = False,
) -> None:
    cfg  = cfg or {}
    want = _lvl(str(cfg.get("log_level") or LOG_LEVEL_DEFAULT))
    have = _lvl(level)
    if have < want:
        return
    stream = sys.stderr if err or level in {"WARN", "ERROR"} else sys.stdout
    print(f"[{level}] {msg}", file=stream, flush=True)


@dataclass
class RunStats:
    t0: float
    last_heartbeat: float
    files:   int = 0
    records: int = 0


def heartbeat(st: RunStats, *, cfg: dict, extra: str = "") -> None:
    every_s = int(cfg.get("log_every_s") or LOG_EVERY_S_DEFAULT)
    now = time.time()
    if now - st.last_heartbeat >= every_s:
        dt   = now - st.t0
        tail = f" {extra}".rstrip()
        log(f"heartbeat t={dt:.0f}s files={st.files} records={st.records}{tail}", cfg=cfg)
        st.last_heartbeat = now


# ── Models ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkRecord:
    id:       str
    text:     str
    metadata: dict


@dataclass(frozen=True)
class Section:
    title: str
    body:  str
    level: int = 0


@dataclass(frozen=True)
class StructuralUnit:
    kind: str
    title: str
    text: str
    level: int = 0
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeResources:
    token_counter:    "TokenCounter"
    semantic_encoder: "SemanticEncoder"


# ── Helpers: hashing, metadata ────────────────────────────────────────────────

def sha256_hex(b: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_chunk_id(
        file_hash: str,
        rel_path: str,
        chunk_index: int,
        chunk_text: str,
) -> str:
    payload = (
        f"{file_hash}|{rel_path}|{chunk_index}"
        f"|{sha256_hex(chunk_text.encode('utf-8', errors='replace'))}"
    )
    return sha256_hex(payload.encode("utf-8"))


def build_metadata(path: Path, rel_path: str, text: str) -> dict:
    st = path.stat()
    return {
        "source_path":     rel_path.replace("\\", "/"),
        "source_name":     path.name,
        "ext":             path.suffix.lower(),
        "size_bytes":      st.st_size,
        "mtime_utc":       datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "ingested_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "text_len":        len(text),
    }


# ── Normalization ─────────────────────────────────────────────────────────────

_WHITESPACE_RE       = re.compile(r"[ \t\f\v]+")
_MULTINEW_RE         = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        line = _WHITESPACE_RE.sub(" ", line)
        lines.append(line)
    text = "\n".join(lines).strip()
    text = _MULTINEW_RE.sub("\n\n", text)
    return text


# ── Tokenization ──────────────────────────────────────────────────────────────

class TokenCounter:
    def __init__(self, cfg: dict):
        self.backend           = str(cfg.get("tokenizer_backend") or TOKENIZER_BACKEND_DEFAULT).strip().lower()
        self.model             = str(cfg.get("tokenizer_model")   or TOKENIZER_MODEL_DEFAULT).strip()
        self._encoder          = None
        self._tokenizer        = None
        self._resolved_backend = "simple"
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend in {"auto", "tiktoken"} and tiktoken is not None:
            try:
                self._encoder = (
                    tiktoken.encoding_for_model(self.model)
                    if self.model and self.model != "cl100k_base"
                    else tiktoken.get_encoding("cl100k_base")
                )
                self._resolved_backend = "tiktoken"
                return
            except Exception:
                pass
        if self.backend in {"auto", "transformers"} and AutoTokenizer is not None and self.model:
            try:
                self._tokenizer        = AutoTokenizer.from_pretrained(self.model, use_fast=True)
                self._resolved_backend = "transformers"
                return
            except Exception:
                pass
        self._resolved_backend = "simple"

    @property
    def resolved_backend(self) -> str:
        return self._resolved_backend

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._resolved_backend == "tiktoken" and self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        if self._resolved_backend == "transformers" and self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


# ── Semantic embeddings ───────────────────────────────────────────────────────

class SemanticEncoder:
    def __init__(self, cfg: dict):
        self.enabled    = bool(cfg.get("enable_semantic_chunking", ENABLE_SEMANTIC_CHUNKING_DEFAULT))
        self.model_name = str(cfg.get("semantic_model") or SEMANTIC_MODEL_DEFAULT).strip()
        self._model     = None
        self.available  = False
        if not self.enabled or SentenceTransformer is None:
            return
        try:
            self._model    = SentenceTransformer(self.model_name)
            self.available = True
        except Exception:
            self.available = False

    def encode(self, texts: list[str]) -> Optional[list[list[float]]]:
        if not self.available or self._model is None or not texts:
            return None
        try:
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [list(map(float, v)) for v in vecs]
        except Exception:
            return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ── Section / sentence splitting ──────────────────────────────────────────────

_MD_HEADER_RE      = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMERIC_HEADER_RE = re.compile(r"^(?:\d+(?:\.\d+){0,4}|[A-Z])[\.\)]\s+.+$")
_SECTION_WORD_RE   = re.compile(
    r"^(?:kapitel|chapter|abschnitt|section|teil|annex|anhang)\b", re.IGNORECASE
)
_BULLET_RE              = re.compile(r"^(?:[-*•]\s+|\d+[\.\)]\s+)")
_SENTENCE_SPLIT_RE      = re.compile(r"(?<=[\.\!\?\:\;])\s+(?=[A-ZÄÖÜ0-9\"„«»])")
_COLON_SUBHEADER_RE     = re.compile(r"^[A-ZÄÖÜ][^.!?]{0,120}:$")
_SHORT_TITLEISH_RE      = re.compile(r"^[A-ZÄÖÜ0-9][^.!?]{0,100}$")
_QUESTION_PREFIX_RE     = re.compile(r"^(?:frage|fragen)\b", re.IGNORECASE)
_CHECKLIST_PREFIX_RE    = re.compile(r"^(?:checkliste|checklist|todo|to-do)\b", re.IGNORECASE)
_LIST_CONTINUATION_RE   = re.compile(r"^(?:[a-zäöü0-9].{0,140}|[A-ZÄÖÜ].{0,140}\?)$")
_PROCEDURE_PREFIX_RE    = re.compile(
    r"^(?:bei |vor |nach |während |zurück |anschliessend |danach |erste\b|weiteres\b)",
    re.IGNORECASE,
)


def looks_like_header(line: str) -> tuple[bool, int, str]:
    s = line.strip()
    if not s:
        return False, 0, ""

    m = _MD_HEADER_RE.match(s)
    if m:
        return True, len(m.group(1)), m.group(2).strip()

    if _SECTION_WORD_RE.match(s):
        return True, 1, s

    if _NUMERIC_HEADER_RE.match(s) and len(s) <= 120:
        return True, 2, s

    if (
            len(s) <= 90
            and not s.endswith((".", "!", "?", ":"))
            and not _BULLET_RE.match(s)
            and len(s.split()) <= 10
    ):
        letters = [c for c in s if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
            titleish    = s == s.title()
            # Require >= 2 words: prevents ALL-CAPS abbreviations like
            # "ENTWURF" or "HINWEIS" from being mistaken for section headers.
            if (upper_ratio > 0.8 and len(s.split()) >= 2) or titleish:
                return True, 3, s

    return False, 0, ""


def looks_like_subheader(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if looks_like_header(s)[0]:
        return False
    if _COLON_SUBHEADER_RE.match(s):
        return True
    if (
            _SHORT_TITLEISH_RE.match(s)
            and len(s.split()) <= 8
            and not _BULLET_RE.match(s)
            and not s.endswith((".", "!", "?"))
    ):
        letters = [c for c in s if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
            if upper_ratio > 0.55 or s == s.title() or _PROCEDURE_PREFIX_RE.match(s):
                return True
    return False


def classify_list_line(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None
    if _BULLET_RE.match(s):
        return "question_block" if "?" in s else "list_block"
    if s.endswith("?"):
        return "question_block"
    if _QUESTION_PREFIX_RE.match(s):
        return "question_block"
    if _CHECKLIST_PREFIX_RE.match(s):
        return "checklist_block"
    return None


def looks_like_list_continuation(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if classify_list_line(s) is not None:
        return True
    return bool(_LIST_CONTINUATION_RE.match(s))


def split_into_sections(text: str, *, enable_section_awareness: bool) -> list[Section]:
    if not text:
        return []
    if not enable_section_awareness:
        return [Section(title="", body=text.strip(), level=0)] if text.strip() else []

    lines         = text.split("\n")
    sections: list[Section] = []
    current_title = ""
    current_level = 0
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_level, current_body
        body = "\n".join(current_body).strip()
        if body:
            sections.append(Section(title=current_title, body=body, level=current_level))
        current_body = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if line and re.fullmatch(r"[-=]{3,}", nxt):
                flush()
                current_title = line
                current_level = 1 if nxt.startswith("=") else 2
                i += 2
                continue
        is_header, level, title = looks_like_header(line)
        if is_header:
            flush()
            current_title = title
            current_level = level
        else:
            current_body.append(line)
        i += 1

    flush()

    if not sections and text.strip():
        sections.append(Section(title="", body=text.strip(), level=0))
    return sections


def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    for para in split_into_paragraphs(text):
        segs = _SENTENCE_SPLIT_RE.split(para)
        if len(segs) == 1:
            segs = re.split(r"(?<=[\.\!\?])\s+", para)
        for seg in segs:
            s = seg.strip()
            if s:
                parts.append(s)
    return parts


def split_section_into_structural_units(section: Section) -> list[StructuralUnit]:
    """
    Split one high-level section into smaller structural units.

    Goals:
      - preserve hierarchical sub-headings (e.g. "Zurück im FOR")
      - keep list/question/checklist blocks together
      - avoid mixing multiple process phases inside one large semantic chunk
    """
    raw_lines = [ln.rstrip() for ln in section.body.split("\n")]
    units: list[StructuralUnit] = []

    current_title = ""
    current_kind = "paragraph"
    current_lines: list[str] = []

    def current_path(title: str) -> tuple[str, ...]:
        parts = [section.title]
        if title:
            parts.append(title)
        return tuple(x for x in parts if x)

    def flush() -> None:
        nonlocal current_title, current_kind, current_lines
        text = "\n".join(x for x in current_lines if x.strip()).strip()
        if text:
            units.append(
                StructuralUnit(
                    kind=current_kind,
                    title=current_title,
                    text=text,
                    level=section.level + (1 if current_title else 0),
                    path=current_path(current_title),
                )
            )
        current_lines = []
        current_kind = "paragraph"

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()

        if not line:
            if current_lines and current_kind == "paragraph":
                current_lines.append("")
            i += 1
            continue

        if looks_like_subheader(line):
            flush()
            current_title = line.rstrip(":").strip()
            i += 1
            continue

        line_kind = classify_list_line(line)

        if line_kind is not None:
            if current_kind not in {"list_block", "question_block", "checklist_block"}:
                flush()
                current_kind = line_kind
            elif current_kind != line_kind and current_lines:
                flush()
                current_kind = line_kind

            current_lines.append(line)

            j = i + 1
            while j < len(raw_lines):
                nxt = raw_lines[j].strip()
                if not nxt:
                    break
                if looks_like_subheader(nxt) or looks_like_header(nxt)[0]:
                    break
                if not looks_like_list_continuation(nxt):
                    break
                current_lines.append(nxt)
                j += 1
            flush()
            i = j
            continue

        if current_kind in {"list_block", "question_block", "checklist_block"}:
            flush()

        current_lines.append(line)
        i += 1

    flush()

    if not units and section.body.strip():
        units.append(
            StructuralUnit(
                kind="paragraph",
                title="",
                text=section.body.strip(),
                level=section.level,
                path=current_path(""),
            )
        )
    return units

# ── Token-based splitting ─────────────────────────────────────────────────────

def split_text_by_words_to_token_limit(
        text: str, *, max_tokens: int, counter: TokenCounter
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if counter.count(text) <= max_tokens:
        return [text]

    words  = text.split()
    pieces: list[str] = []
    cur:    list[str] = []

    for word in words:
        candidate = " ".join(cur + [word]).strip()
        if cur and counter.count(candidate) > max_tokens:
            pieces.append(" ".join(cur).strip())
            cur = [word]
        else:
            cur.append(word)
    if cur:
        pieces.append(" ".join(cur).strip())

    out: list[str] = []
    for piece in pieces:
        if counter.count(piece) <= max_tokens:
            out.append(piece)
            continue
        raw   = piece
        step  = max(50, len(raw) // 8)
        start = 0
        while start < len(raw):
            end  = min(len(raw), start + step)
            best = raw[start:end]
            while end < len(raw) and counter.count(best) <= max_tokens:
                end  = min(len(raw), end + step)
                nxt  = raw[start:end]
                if counter.count(nxt) > max_tokens:
                    break
                best = nxt
            out.append(best.strip())
            start += max(1, len(best))

    return [x for x in out if x]


def split_unit_to_fit(
        text: str, *, max_tokens: int, counter: TokenCounter
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if counter.count(text) <= max_tokens:
        return [text]

    sents = split_into_sentences(text)
    if len(sents) > 1:
        units: list[str] = []
        cur:   list[str] = []
        for sent in sents:
            candidate = "\n".join(cur + [sent]).strip()
            if cur and counter.count(candidate) > max_tokens:
                units.append("\n".join(cur).strip())
                cur = [sent]
            else:
                cur.append(sent)
        if cur:
            units.append("\n".join(cur).strip())
        out: list[str] = []
        for unit in units:
            if counter.count(unit) <= max_tokens:
                out.append(unit)
            else:
                out.extend(
                    split_text_by_words_to_token_limit(unit, max_tokens=max_tokens, counter=counter)
                )
        return [x for x in out if x]

    return split_text_by_words_to_token_limit(text, max_tokens=max_tokens, counter=counter)


def last_units_with_overlap(
        units: list[str], *, overlap_tokens: int, counter: TokenCounter
) -> list[str]:
    if overlap_tokens <= 0 or not units:
        return []
    acc:   list[str] = []
    total: int       = 0
    for unit in reversed(units):
        t = counter.count(unit)
        if acc and total + t > overlap_tokens:
            break
        acc.append(unit)
        total += t
        if total >= overlap_tokens:
            break
    acc.reverse()
    return acc


# ── Semantic chunking ─────────────────────────────────────────────────────────

def prepare_semantic_units(
        text: str, *, chunk_size_tokens: int, counter: TokenCounter
) -> list[str]:
    paras = split_into_paragraphs(text)
    if not paras:
        return []
    target = max(40, min(140, chunk_size_tokens // 2))
    units: list[str] = []

    for para in paras:
        para = para.strip()
        if not para:
            continue
        if counter.count(para) <= target:
            units.append(para)
            continue
        sents = split_into_sentences(para)
        if not sents:
            units.extend(split_unit_to_fit(para, max_tokens=target, counter=counter))
            continue
        cur: list[str] = []
        for sent in sents:
            candidate = " ".join(cur + [sent]).strip()
            if cur and counter.count(candidate) > target:
                units.append(" ".join(cur).strip())
                cur = [sent]
            else:
                cur.append(sent)
        if cur:
            units.append(" ".join(cur).strip())

    out: list[str] = []
    for unit in units:
        out.extend(split_unit_to_fit(unit, max_tokens=target, counter=counter))
    return [u for u in out if u.strip()]


def _structural_prefix(unit: StructuralUnit) -> str:
    if unit.path:
        return "\n\n".join(x for x in unit.path if x).strip()
    if unit.title:
        return unit.title.strip()
    return ""


def semantic_chunk_structural_unit(
        unit: StructuralUnit,
        *,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        min_chunk_tokens: int,
        counter: TokenCounter,
        semantic_encoder: SemanticEncoder,
        semantic_threshold: float,
        semantic_min_units: int,
) -> list[tuple[str, dict]]:
    """
    Chunk one structural unit. For paragraphs we still use semantic chunking.
    For question/list/checklist blocks we preserve structure and split conservatively.
    """
    header_prefix = _structural_prefix(unit)
    header_prefix = f"{header_prefix}\n\n" if header_prefix else ""
    header_tokens = counter.count(header_prefix)
    max_body = max(1, chunk_size_tokens - header_tokens)

    if unit.kind in {"list_block", "question_block", "checklist_block"}:
        pieces = split_unit_to_fit(unit.text, max_tokens=max_body, counter=counter)
        chunks: list[tuple[str, dict]] = []
        for piece in pieces:
            text = f"{header_prefix}{piece}".strip()
            tok_len = counter.count(text)
            if tok_len < min_chunk_tokens and not chunks:
                # Preserve short but structurally important list/question blocks.
                pass
            elif tok_len < min_chunk_tokens:
                continue
            chunks.append((text, {
                "section_title": unit.path[0] if unit.path else unit.title,
                "subsection_title": unit.path[-1] if len(unit.path) > 1 else unit.title,
                "section_level": unit.level,
                "content_type": unit.kind,
                "hierarchy_path": list(unit.path),
                "semantic_chunking_used": False,
                "semantic_similarity_prev": None,
            }))
        return chunks

    units = prepare_semantic_units(unit.text, chunk_size_tokens=max_body, counter=counter)
    if not units:
        full = f"{header_prefix}{unit.text}".strip()
        if full and counter.count(full) >= min_chunk_tokens:
            return [(full, {
                "section_title": unit.path[0] if unit.path else unit.title,
                "subsection_title": unit.path[-1] if len(unit.path) > 1 else unit.title,
                "section_level": unit.level,
                "content_type": unit.kind,
                "hierarchy_path": list(unit.path),
                "semantic_chunking_used": False,
                "semantic_similarity_prev": None,
            })]
        return []

    embeddings = semantic_encoder.encode(units)
    use_semantic = embeddings is not None and len(embeddings) == len(units)

    chunks: list[tuple[str, dict]] = []
    current_units: list[str] = []
    current_sims: list[float] = []

    def flush() -> None:
        nonlocal current_units, current_sims
        if not current_units:
            return
        body = "\n\n".join(current_units).strip()
        text = f"{header_prefix}{body}".strip()
        tok_len = counter.count(text)
        if tok_len >= min_chunk_tokens or not chunks:
            avg_sim = sum(current_sims) / len(current_sims) if current_sims else None
            chunks.append((text, {
                "section_title": unit.path[0] if unit.path else unit.title,
                "subsection_title": unit.path[-1] if len(unit.path) > 1 else unit.title,
                "section_level": unit.level,
                "content_type": unit.kind,
                "hierarchy_path": list(unit.path),
                "semantic_chunking_used": use_semantic,
                "semantic_similarity_prev": avg_sim,
            }))
        overlap_units = last_units_with_overlap(
            current_units, overlap_tokens=chunk_overlap_tokens, counter=counter
        )
        current_units = overlap_units[:]
        current_sims = []

    for idx, unit_text in enumerate(units):
        if not current_units:
            current_units = [unit_text]
            continue

        candidate_body = "\n\n".join(current_units + [unit_text]).strip()
        candidate_text = f"{header_prefix}{candidate_body}".strip()
        candidate_tokens = counter.count(candidate_text)

        sim = None
        if use_semantic and idx > 0:
            sim = cosine_similarity(embeddings[idx - 1], embeddings[idx])

        should_split = (
                use_semantic
                and len(current_units) >= semantic_min_units
                and sim is not None
                and sim < semantic_threshold
                and counter.count(f"{header_prefix}{' '.join(current_units)}") >= min_chunk_tokens
        )

        if candidate_tokens > chunk_size_tokens:
            flush()
            if not current_units:
                current_units = [unit_text]
            else:
                merged = "\n\n".join(current_units + [unit_text]).strip()
                if counter.count(f"{header_prefix}{merged}") > chunk_size_tokens:
                    pieces = split_unit_to_fit(unit_text, max_tokens=max_body, counter=counter)
                    for piece in pieces:
                        cand = "\n\n".join(current_units + [piece]).strip()
                        if counter.count(f"{header_prefix}{cand}") > chunk_size_tokens:
                            flush()
                        current_units.append(piece)
                        if counter.count(
                                f"{header_prefix}{' '.join(current_units)}"
                        ) >= chunk_size_tokens:
                            flush()
            continue

        if should_split:
            flush()

        current_units.append(unit_text)
        if sim is not None:
            current_sims.append(sim)

    flush()

    if len(chunks) >= 2:
        last_text, last_meta = chunks[-1]
        if counter.count(last_text) < min_chunk_tokens:
            prev_text, prev_meta = chunks[-2]
            merged = f"{prev_text}\n\n{last_text}".strip()
            if counter.count(merged) <= chunk_size_tokens + chunk_overlap_tokens:
                merged_meta = dict(prev_meta)
                merged_meta["merged_small_tail"] = True
                chunks[-2] = (merged, merged_meta)
                chunks.pop()

    return chunks


def chunk_structural_units(
        units: list[StructuralUnit],
        *,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        min_chunk_tokens: int,
        counter: TokenCounter,
        semantic_encoder: SemanticEncoder,
        semantic_threshold: float,
        semantic_min_units: int,
) -> list[tuple[str, dict]]:
    chunks: list[tuple[str, dict]] = []
    for unit in units:
        chunks.extend(
            semantic_chunk_structural_unit(
                unit,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                min_chunk_tokens=min_chunk_tokens,
                counter=counter,
                semantic_encoder=semantic_encoder,
                semantic_threshold=semantic_threshold,
                semantic_min_units=semantic_min_units,
            )
        )
    return chunks


def structural_token_chunk_text(
        text: str,
        *,
        cfg: dict,
        resources: RuntimeResources,
) -> Iterator[Tuple[int, str, dict]]:
    chunk_size_tokens    = int(cfg.get("chunk_size_tokens")    or CHUNK_SIZE_TOKENS_DEFAULT)
    chunk_overlap_tokens = int(cfg.get("chunk_overlap_tokens") or CHUNK_OVERLAP_TOKENS_DEFAULT)
    min_chunk_tokens     = int(cfg.get("min_chunk_tokens")     or MIN_CHUNK_TOKENS_DEFAULT)

    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    counter        = resources.token_counter
    sem_encoder    = resources.semantic_encoder
    enable_section = bool(cfg.get("enable_section_awareness", ENABLE_SECTION_AWARENESS_DEFAULT))
    sem_threshold  = float(cfg.get("semantic_threshold") or SEMANTIC_THRESHOLD_DEFAULT)
    sem_min_units  = int(cfg.get("semantic_min_units")   or SEMANTIC_MIN_UNITS_DEFAULT)

    sections  = split_into_sections(text, enable_section_awareness=enable_section)
    chunk_idx = 0

    for section_index, section in enumerate(sections):
        structural_units = split_section_into_structural_units(section)
        chunks = chunk_structural_units(
            structural_units,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            min_chunk_tokens=min_chunk_tokens,
            counter=counter,
            semantic_encoder=sem_encoder,
            semantic_threshold=sem_threshold,
            semantic_min_units=sem_min_units,
        )
        for chunk_text, extra_meta in chunks:
            effective_min = min_chunk_tokens
            if extra_meta.get("content_type") in {"list_block", "question_block", "checklist_block"}:
                effective_min = max(20, min_chunk_tokens // 2)
            if counter.count(chunk_text) < effective_min:
                continue
            meta = {
                "section_index":              section_index,
                "section_title":              extra_meta.get("section_title", ""),
                "subsection_title":           extra_meta.get("subsection_title", ""),
                "section_level":              extra_meta.get("section_level", 0),
                "content_type":               extra_meta.get("content_type", "paragraph"),
                "hierarchy_path":             extra_meta.get("hierarchy_path", []),
                "semantic_chunking_used":     extra_meta.get("semantic_chunking_used", False),
                "semantic_similarity_prev":   extra_meta.get("semantic_similarity_prev"),
                "tokenizer_backend_resolved": counter.resolved_backend,
            }
            if "merged_small_tail" in extra_meta:
                meta["merged_small_tail"] = extra_meta["merged_small_tail"]
            if "split_from_long_unit" in extra_meta:
                meta["split_from_long_unit"] = extra_meta["split_from_long_unit"]
            yield chunk_idx, chunk_text, meta
            chunk_idx += 1


# ── Plain-text reader ─────────────────────────────────────────────────────────

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


# ── DOCX reader ───────────────────────────────────────────────────────────────

def read_docx(path: Path) -> str:
    if Document is None:
        raise RuntimeError("Missing dependency: python-docx (pip install python-docx)")

    doc   = Document(str(path))
    parts = list(_iter_block_text(doc))

    cleaned: list[str] = []
    prev_blank = False
    for part in parts:
        s = _normalize_docx_text(part)
        if not s:
            if not prev_blank and cleaned:
                cleaned.append("")
            prev_blank = True
            continue
        cleaned.append(s)
        prev_blank = False

    return "\n".join(cleaned).strip()


def _iter_block_text(
        parent: "_Document | _Cell", *, table_depth: int = 0
) -> Iterator[str]:
    for block in _iter_block_items(parent):
        if isinstance(block, Paragraph):
            text = _paragraph_text(block)
            if text:
                yield text
        elif isinstance(block, Table):
            table_text = _table_to_text(block, table_depth=table_depth)
            if table_text:
                yield table_text


def _iter_block_items(
        parent: "_Document | _Cell",
) -> "Iterator[Paragraph | Table]":
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise TypeError(f"Unsupported parent type: {type(parent)!r}")
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _paragraph_text(paragraph: "Paragraph") -> str:
    return _normalize_docx_text(paragraph.text)


def _table_to_text(table: "Table", *, table_depth: int = 0) -> str:
    row_lines: list[str]       = []
    seen: set[tuple[str, ...]] = set()

    for row in table.rows:
        cell_texts = [_cell_text(cell, table_depth=table_depth + 1) for cell in row.cells]
        sig = tuple(cell_texts)
        if not any(t.strip() for t in cell_texts):
            continue
        if sig in seen:
            continue
        seen.add(sig)
        line = _normalize_docx_text(" | ".join(cell_texts))
        if line:
            row_lines.append(line)

    return "\n".join(row_lines)


def _cell_text(cell: "_Cell", *, table_depth: int) -> str:
    parts: list[str] = []
    seen:  set[str]  = set()
    for item in _iter_block_text(cell, table_depth=table_depth):
        s = _normalize_docx_text(item)
        if not s or s in seen:
            continue
        seen.add(s)
        parts.append(s)
    return " // ".join(parts).strip()


def _normalize_docx_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.strip().split())
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


# ── OCR helpers (Tesseract) ───────────────────────────────────────────────────

def _resolve_tesseract_cmd(explicit: Optional[str]) -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found
        raise RuntimeError(f"TESSERACT_CMD set but not found: {explicit}")
    found = shutil.which("tesseract")
    if found:
        return found
    raise RuntimeError("tesseract not found in PATH; set TESSERACT_CMD.")


def _resolve_tessdata_dir(
        tesseract_cmd: str, explicit_dir: Optional[str]
) -> Optional[str]:
    if explicit_dir:
        p = Path(explicit_dir)
        if p.exists():
            return str(p)
        raise RuntimeError(f"TESSDATA_DIR set but does not exist: {explicit_dir}")
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix:
        pref = Path(prefix).expanduser()
        if pref.name == "tessdata" and pref.exists():
            return str(pref)
        cand = pref / "tessdata"
        if cand.exists():
            return str(cand)
    try:
        r = subprocess.run(
            [str(tesseract_cmd), "--print-tessdata-dir"],
            capture_output=True, text=True, check=False,
        )
        out = (r.stdout or "").strip()
        if out and Path(out).exists():
            return out
    except Exception:
        pass
    return None


def _ocr_fitz_page(
        page: "fitz.Page",
        *,
        lang: str,
        tessdata_dir: Optional[str],
        dpi: int = 300,
) -> str:
    """
    Render a PyMuPDF page to a temp PNG and run Tesseract OCR on it.

    Uses a temporary file instead of PIL so that Pillow is not required.
    Only called for Type-B pages (scanned, no text layer).
    """
    if pytesseract is None or fitz is None:
        return ""
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        pix.save(tmp_path)

        try:
            tcfg   = f"--tessdata-dir {tessdata_dir}" if tessdata_dir else ""
            result = (pytesseract.image_to_string(tmp_path, lang=lang, config=tcfg) or "").strip()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return result
    except Exception:
        return ""


# ── PyMuPDF image-extraction helpers ─────────────────────────────────────────

def _image_cache_path(
        source_path: str,
        page_index: int,
        img_index: int,
        ext: str,
        cache_dir: Path,
) -> Path:
    """
    Deterministic, filesystem-safe cache path for one extracted image.

    Format: <cache_dir>/<safe_source>_p<page>_i<img>.<ext>

    Separators and special characters in source_path are replaced with
    underscores so the filename stays flat and portable.
    """
    safe = re.sub(r"[/\\:*?\"<>|]", "_", source_path)
    return cache_dir / f"{safe}_p{page_index}_i{img_index}.{ext}"


def classify_pdf_page(
        doc: "fitz.Document",
        page: "fitz.Page",
        *,
        min_text_chars: int = 80,
        full_page_image_ratio: float = 0.85,
) -> dict:
    """
    Inspect one PDF page and classify what it contains.

    Three page types exist in practice:

      Type A — text PDF with embedded figures
        get_text() returns substantial text; images are smaller than the page.
        Action: extract text normally; save embedded images to cache.

      Type B — scanned PDF (the page IS a raster image, no text layer)
        get_text() returns almost nothing; one large image covers the page.
        Action: run OCR.

      Type C — searchable scan (OCR layer already present in the PDF)
        get_text() returns text; one large image covers the page.
        Action: use existing text layer; skip OCR.

    The page type is determined by comparing each embedded image's area
    (converted to PDF points) against the full page area.

    Returns:
        text              selectable text extracted by PyMuPDF
        needs_ocr         True only for Type B (scan without a text layer)
        embedded_images   list of xrefs for non-full-page figures (Type A)
        is_full_page_scan True for Type B and Type C
    """
    text      = page.get_text().strip()
    page_area = page.rect.width * page.rect.height

    embedded_images: list[int] = []
    is_full_page_scan           = False

    for img in page.get_images(full=True):
        xref = img[0]
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue

        img_w = base["width"]
        img_h = base["height"]
        xres  = base.get("xres") or 72
        yres  = base.get("yres") or 72

        # Convert image dimensions from pixels → PDF points (1 pt = 1/72 inch).
        # page.rect is in points; without this conversion coverage would be
        # computed in mixed units and give nonsensical results.
        img_w_pt = img_w * 72.0 / xres
        img_h_pt = img_h * 72.0 / yres
        coverage = (img_w_pt * img_h_pt) / page_area if page_area > 0 else 0.0

        if coverage >= full_page_image_ratio:
            is_full_page_scan = True      # image IS the page
        else:
            embedded_images.append(xref)  # image is a figure within the page

    # OCR is only necessary for Type B.
    # Type C already has a usable text layer returned by get_text().
    needs_ocr = is_full_page_scan and len(text) < min_text_chars

    return {
        "text":              text,
        "needs_ocr":         needs_ocr,
        "embedded_images":   embedded_images,
        "is_full_page_scan": is_full_page_scan,
    }


def extract_page_images(
        doc: "fitz.Document",
        *,
        xrefs: list[int],
        page_index: int,
        source_path: str,
        cache_dir: Path,
        min_px: int = 100,
) -> list[str]:
    """
    Extract embedded figures from a PDF page and write them to the image cache.

    Skips images smaller than min_px in either dimension — these are typically
    decorative elements (icons, rule lines, bullets) with no semantic value.

    Returns absolute file paths of all saved images.
    Paths are stored in chunk metadata so the inference pipeline can find
    them for lazy vision captioning without re-parsing the PDF.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for img_index, xref in enumerate(xrefs):
        try:
            base      = doc.extract_image(xref)
            img_bytes = base["image"]
            ext       = base.get("ext", "png")
            w         = base.get("width",  0)
            h         = base.get("height", 0)

            if w < min_px or h < min_px:
                continue

            dest = _image_cache_path(source_path, page_index, img_index, ext, cache_dir)
            dest.write_bytes(img_bytes)
            saved.append(str(dest))
        except Exception:
            continue

    return saved


# ── pdfplumber per-page text helper ──────────────────────────────────────────

def _extract_pdfplumber_page_text(page) -> str:
    """
    Extract text and table content from a single pdfplumber page.

    pdfplumber recovers table cell content by analysing character positions,
    which significantly outperforms get_text() on complex grid layouts.
    On the reference document this yields 24 k chars vs PyMuPDF's 11 k.

    Tables are formatted as pipe-separated rows and appended after the
    page's running text so both are present in the same chunk.

    Returns an empty string if both text and tables are empty.
    """
    try:
        page_text = (page.extract_text() or "").strip()

        table_rows: list[str] = []
        for table in (page.extract_tables() or []):
            for row in table:
                cleaned = [str(cell or "").strip() for cell in row]
                if any(cleaned):
                    table_rows.append(" | ".join(cleaned))

        if table_rows:
            table_block = "\n".join(table_rows)
            return f"{page_text}\n\n{table_block}".strip() if page_text else table_block

        return page_text
    except Exception:
        return ""


# ── PDF reader (hybrid: pdfplumber text + PyMuPDF images) ────────────────────

def read_pdf(path: Path, *, cfg: dict) -> Tuple[str, dict]:
    """
    Hybrid PDF reader.

    Text extraction per page (in priority order):
      1. pdfplumber  — primary; best layout and table recovery (verified on
                       real documents: ~2× more chars than PyMuPDF alone)
      2. PyMuPDF     — automatic fallback if pdfplumber is unavailable or
                       raises an exception on a specific file
      3. Tesseract   — OCR fallback for scanned pages without a text layer
                       (triggered only when the page is classified as Type B)

    Image extraction always uses PyMuPDF:
      - classify_pdf_page() distinguishes embedded figures from full-page scans
      - extract_page_images() saves figures to the image cache folder
      - Image paths are stored in metadata["embedded_images"] for the
        inference pipeline to use during lazy vision captioning

    The metadata field "pdf_text_reader" records which library produced the
    final text so results can be audited and reproduced.
    """
    if pdfplumber is None and fitz is None:
        raise RuntimeError(
            "At least one PDF library is required: "
            "pdfplumber (pip install pdfplumber) or PyMuPDF (pip install PyMuPDF)"
        )

    enable_ocr    = bool (cfg.get("enable_ocr",               ENABLE_OCR_DEFAULT))
    ocr_lang      = str  (cfg.get("ocr_lang")              or OCR_LANG_DEFAULT)
    ocr_min_chars = int  (cfg.get("ocr_min_chars_per_page") or OCR_MIN_CHARS_PER_PAGE_DEFAULT)
    ocr_dpi       = int  (cfg.get("ocr_dpi")               or OCR_DPI_DEFAULT)
    img_ratio     = float(cfg.get("full_page_image_ratio")  or FULL_PAGE_IMAGE_RATIO_DEFAULT)
    min_px        = int  (cfg.get("min_image_px")           or MIN_IMAGE_PX_DEFAULT)
    cache_dir     = Path (str(cfg.get("image_cache_dir")    or IMAGE_CACHE_DIR_DEFAULT))

    meta: dict = {
        "pdf_ocr_used":    False,
        "pdf_pages":       0,
        "pdf_text_chars":  0,
        "pdf_text_reader": None,  # "pdfplumber" | "pymupdf" | either + "+ocr"
        "ocr_lang":        ocr_lang,
        "embedded_images": [],    # paths of all figures saved to image cache
    }

    # Tesseract is resolved lazily — only when a page actually needs OCR
    _tess_cmd:  Optional[str] = None
    _tess_data: Optional[str] = None

    def _ensure_tesseract() -> Optional[str]:
        nonlocal _tess_cmd, _tess_data
        if _tess_cmd is None:
            _tess_cmd  = _resolve_tesseract_cmd(cfg.get("tesseract_cmd"))
            _tess_data = _resolve_tessdata_dir(_tess_cmd, cfg.get("tessdata_dir"))
            if pytesseract is not None:
                pytesseract.pytesseract.tesseract_cmd = _tess_cmd
        return _tess_data

    # Open PyMuPDF once for the whole file.
    # Used for image extraction and OCR rendering regardless of which
    # library handles text extraction.
    fitz_doc = None
    if fitz is not None:
        try:
            fitz_doc = fitz.open(str(path))
        except Exception as e:
            log(
                f"PyMuPDF failed to open {path.name}: {e} — image extraction disabled",
                level="WARN", cfg=cfg, err=True,
            )

    parts: list[str]           = []
    all_image_paths: list[str] = []
    use_pdfplumber             = pdfplumber is not None

    # ── Branch A: pdfplumber (primary text path) ──────────────────────────────
    if use_pdfplumber:
        try:
            with pdfplumber.open(path) as plumber_doc:
                meta["pdf_pages"]       = len(plumber_doc.pages)
                meta["pdf_text_reader"] = "pdfplumber"

                for page_idx, plumber_page in enumerate(plumber_doc.pages):
                    page_text = _extract_pdfplumber_page_text(plumber_page)

                    if fitz_doc is not None:
                        fitz_page = fitz_doc.load_page(page_idx)
                        info      = classify_pdf_page(
                            fitz_doc, fitz_page,
                            min_text_chars=ocr_min_chars,
                            full_page_image_ratio=img_ratio,
                        )

                        # Save embedded figures to the image cache
                        if info["embedded_images"]:
                            saved = extract_page_images(
                                fitz_doc,
                                xrefs=info["embedded_images"],
                                page_index=page_idx,
                                source_path=str(path),
                                cache_dir=cache_dir,
                                min_px=min_px,
                            )
                            all_image_paths.extend(saved)

                        # OCR: only when pdfplumber text is sparse AND PyMuPDF
                        # confirms the page is a scan (not just low-content text)
                        if (
                                len(page_text) < ocr_min_chars
                                and info["is_full_page_scan"]
                                and enable_ocr
                        ):
                            tessdata  = _ensure_tesseract()
                            ocr_text  = _ocr_fitz_page(
                                fitz_page, lang=ocr_lang,
                                tessdata_dir=tessdata, dpi=ocr_dpi,
                            )
                            if ocr_text:
                                page_text = ocr_text
                                meta["pdf_ocr_used"]    = True
                                meta["pdf_text_reader"] = "pdfplumber+ocr"

                    if page_text:
                        parts.append(page_text)

        except Exception as e:
            # pdfplumber crashed on this specific file → fall through to Branch B
            log(
                f"pdfplumber failed on {path.name}: {e} — falling back to PyMuPDF",
                level="WARN", cfg=cfg, err=True,
            )
            parts          = []
            use_pdfplumber = False  # signal Branch B to run

    # ── Branch B: PyMuPDF only (fallback when pdfplumber unavailable/crashed) ─
    if not use_pdfplumber or not parts:
        if fitz_doc is not None:
            meta["pdf_pages"]       = len(fitz_doc)
            meta["pdf_text_reader"] = "pymupdf"

            for fitz_page in fitz_doc:
                info      = classify_pdf_page(
                    fitz_doc, fitz_page,
                    min_text_chars=ocr_min_chars,
                    full_page_image_ratio=img_ratio,
                )
                page_text = info["text"]

                if info["embedded_images"]:
                    saved = extract_page_images(
                        fitz_doc,
                        xrefs=info["embedded_images"],
                        page_index=fitz_page.number,
                        source_path=str(path),
                        cache_dir=cache_dir,
                        min_px=min_px,
                    )
                    all_image_paths.extend(saved)

                if info["needs_ocr"] and enable_ocr:
                    tessdata  = _ensure_tesseract()
                    ocr_text  = _ocr_fitz_page(
                        fitz_page, lang=ocr_lang,
                        tessdata_dir=tessdata, dpi=ocr_dpi,
                    )
                    if ocr_text:
                        page_text = ocr_text
                        meta["pdf_ocr_used"]    = True
                        meta["pdf_text_reader"] = "pymupdf+ocr"

                if page_text:
                    parts.append(page_text)

    if fitz_doc is not None:
        fitz_doc.close()

    full_text               = "\n\n".join(parts).strip()
    meta["pdf_text_chars"]  = len(full_text)
    meta["embedded_images"] = all_image_paths
    return full_text, meta


# ── Dispatch: read any supported format ──────────────────────────────────────

def read_any(path: Path, *, rel_path: str, cfg: dict) -> Tuple[str, dict]:
    """
    Return (text, read_meta) for any supported file type.

    PPTX is converted to PDF via LibreOffice first, then processed with
    read_pdf(). The PPTX-specific OCR flag overrides the general OCR flag
    only for that conversion step.
    """
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return read_text_file(path), {}

    if ext == ".docx":
        return read_docx(path), {}

    if ext == ".pdf":
        return read_pdf(path, cfg=cfg)

    if ext == ".pptx":
        pdf_path = pptx_to_pdf(path, cfg.get("soffice_cmd"), cfg_for_log=cfg)
        pptx_cfg = {**cfg, "enable_ocr": bool(cfg.get("enable_ocr_pptx", False))}
        pdf_text, pdf_meta = read_pdf(pdf_path, cfg=pptx_cfg)
        pdf_meta.update({
            "origin_source_path": rel_path.replace("\\", "/"),
            "origin_source_name": path.name,
            "origin_ext":         ".pptx",
            "converted_pdf":      pdf_path.name,
        })
        return pdf_text, pdf_meta

    raise ValueError(f"Unsupported extension: {ext}")


# ── LibreOffice PPTX → PDF conversion ────────────────────────────────────────

def resolve_soffice_cmd(explicit: Optional[str]) -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found
        raise RuntimeError(f"SOFFICE_CMD given but not found: {explicit}")
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    if sys.platform == "darwin":
        for candidate in [
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
            Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]:
            if candidate.exists():
                return str(candidate)
    raise RuntimeError(
        "LibreOffice (soffice) not found. Install LibreOffice or set SOFFICE_CMD."
    )


def pptx_to_pdf(path: Path, soffice_cmd: Optional[str], *, cfg_for_log: dict) -> Path:
    cmd     = resolve_soffice_cmd(soffice_cmd)
    out_dir = path.parent / ".converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"{path.stem}.pdf"

    if out_pdf.exists() and out_pdf.stat().st_mtime >= path.stat().st_mtime:
        return out_pdf

    log(f"PPTX→PDF convert start file={path.name}", cfg=cfg_for_log)
    subprocess.run(
        [
            str(cmd), "--headless", "--nologo", "--nolockcheck",
            "--nodefault", "--norestore",
            "--convert-to", "pdf", "--outdir", str(out_dir), str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    if not out_pdf.exists():
        raise RuntimeError(f"PPTX→PDF failed: expected {out_pdf}")
    return out_pdf


# ── Vision utility – INFERENCE TIME only, never called during ingestion ───────

def ollama_caption_png(
        *,
        cfg: dict,
        png_bytes: bytes,
        prompt: Optional[str] = None,
) -> str:
    """
    Send a PNG image to qwen2.5vl:7b via the Ollama /api/chat endpoint
    and return the model's description.

    This is an INFERENCE-TIME utility. Import and call it from your query /
    error-detection pipeline after retrieving a chunk whose metadata contains
    paths in "embedded_images".

    Args:
        cfg:       config dict — must contain "vision_model" and
                   "ollama_base_url"; optionally "vision_timeout_s"
        png_bytes: raw PNG bytes of the image to describe
        prompt:    override prompt; falls back to VISION_PROMPT_FIGURE_DEFAULT
    """
    try:
        import requests  # pip install requests
    except Exception as e:
        raise RuntimeError("Missing dependency: requests (pip install requests)") from e

    base_url = str(cfg.get("ollama_base_url") or OLLAMA_BASE_URL_DEFAULT).rstrip("/")
    model    = str(cfg.get("vision_model")    or "").strip()
    if not model:
        raise RuntimeError("VISION_MODEL is required for vision captioning (e.g. qwen2.5vl:7b)")

    effective_prompt = (
            prompt or str(cfg.get("vision_prompt") or VISION_PROMPT_FIGURE_DEFAULT)
    ).strip()
    timeout_s = int(cfg.get("vision_timeout_s") or VISION_TIMEOUT_S_DEFAULT)
    b64       = base64.b64encode(png_bytes).decode("ascii")

    payload = {
        "model":  model,
        "stream": False,
        "messages": [
            {
                "role":    "system",
                "content": "Du bist ein präziser Assistent für technische Dokumentenanalyse.",
            },
            {
                "role":    "user",
                "content": effective_prompt,
                "images":  [b64],
            },
        ],
    }

    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
    r.raise_for_status()
    content = (r.json().get("message") or {}).get("content")
    return content.strip() if isinstance(content, str) else ""


# ── Record generation ─────────────────────────────────────────────────────────

def iter_records_for_path(
        path: Path,
        *,
        data_dir: Path,
        cfg: dict,
        st: RunStats,
        resources: RuntimeResources,
) -> Iterator[ChunkRecord]:
    """
    Yield ChunkRecord objects for a single file.

    All file types go through the same pipeline:
        read_any() → normalize_text() → structural_token_chunk_text()

    Image paths discovered by read_pdf() are carried in base_meta and
    inherited by every chunk from that file. The inference pipeline uses
    chunk.metadata["embedded_images"] to locate figures for lazy captioning.
    """
    rel_path = str(path.relative_to(data_dir)).replace("\\", "/")

    raw, read_meta = read_any(path, rel_path=rel_path, cfg=cfg)
    text = normalize_text(raw)
    if not text:
        log(f"skip (no text) file={rel_path}", level="DEBUG", cfg=cfg)
        return

    fhash     = file_sha256(path)
    base_meta = build_metadata(path, rel_path, text)
    base_meta.update({"file_sha256": fhash, **read_meta})

    chunk_size_tokens    = int(cfg.get("chunk_size_tokens")    or CHUNK_SIZE_TOKENS_DEFAULT)
    chunk_overlap_tokens = int(cfg.get("chunk_overlap_tokens") or CHUNK_OVERLAP_TOKENS_DEFAULT)
    min_chunk_tokens     = int(cfg.get("min_chunk_tokens")     or MIN_CHUNK_TOKENS_DEFAULT)

    for chunk_index, ctext, extra_meta in structural_token_chunk_text(
            text, cfg=cfg, resources=resources
    ):
        heartbeat(st, cfg=cfg, extra=f"current={rel_path} chunk={chunk_index}")
        meta = {
            **base_meta,
            "chunk_index":           chunk_index,
            "chunk_len":             len(ctext),
            "chunk_size_tokens":     chunk_size_tokens,
            "chunk_overlap_tokens":  chunk_overlap_tokens,
            "min_chunk_tokens":      min_chunk_tokens,
            **extra_meta,
        }
        yield ChunkRecord(
            id=stable_chunk_id(fhash, base_meta["source_path"], chunk_index, ctext),
            text=ctext,
            metadata=meta,
        )


# ── Ingestion loop ────────────────────────────────────────────────────────────

def iter_files(data_dir: Path) -> Iterator[Path]:
    for p in data_dir.rglob("*"):
        # Skip the LibreOffice conversion cache entirely
        if ".converted" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p

def ingest(
        data_dir: Path,
        out_path: Path,
        *,
        cfg: dict,
        resources: RuntimeResources,
) -> Tuple[int, int]:
    st = RunStats(t0=time.time(), last_heartbeat=time.time())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files_processed = records_written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for path in iter_files(data_dir):
            files_processed += 1
            st.files = files_processed
            rel = str(path.relative_to(data_dir)).replace("\\", "/")
            log(f"ingest start file={rel}", cfg=cfg)

            try:
                wrote = 0
                for rec in iter_records_for_path(
                        path, data_dir=data_dir, cfg=cfg, st=st, resources=resources
                ):
                    out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    records_written += 1
                    wrote += 1
                    st.records = records_written
                    heartbeat(st, cfg=cfg, extra=f"current={rel}")
                log(f"ingest done file={rel} records={wrote}", cfg=cfg)

            except Exception as e:
                log(f"ingest failed file={rel}: {e}", level="WARN", cfg=cfg, err=True)

    return files_processed, records_written


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Ingest ./data/** → section-aware, semantic-chunked JSONL.\n"
            "Text:   pdfplumber (primary) + PyMuPDF fallback.\n"
            "Images: PyMuPDF extraction to image cache (lazy captioning at inference time)."
        )
    )

    ap.add_argument("--data_dir",        type=str, default=DATA_DIR_DEFAULT)
    ap.add_argument("--out",             type=str, default=OUT_JSONL_DEFAULT)
    ap.add_argument("--image_cache_dir", type=str, default=IMAGE_CACHE_DIR_DEFAULT,
                    help="Folder where extracted PDF images are saved for inference-time captioning")

    ap.add_argument("--chunk_size_tokens",    type=int, default=CHUNK_SIZE_TOKENS_DEFAULT)
    ap.add_argument("--chunk_overlap_tokens", type=int, default=CHUNK_OVERLAP_TOKENS_DEFAULT)
    ap.add_argument("--min_chunk_tokens",     type=int, default=MIN_CHUNK_TOKENS_DEFAULT)

    ap.add_argument("--tokenizer_backend", type=str, default=TOKENIZER_BACKEND_DEFAULT,
                    help="auto|tiktoken|transformers|simple")
    ap.add_argument("--tokenizer_model",   type=str, default=TOKENIZER_MODEL_DEFAULT)

    ap.add_argument("--enable_section_awareness", action="store_true",
                    default=ENABLE_SECTION_AWARENESS_DEFAULT)
    ap.add_argument("--enable_semantic_chunking", action="store_true",
                    default=ENABLE_SEMANTIC_CHUNKING_DEFAULT)
    ap.add_argument("--semantic_model",     type=str,   default=SEMANTIC_MODEL_DEFAULT)
    ap.add_argument("--semantic_threshold", type=float, default=SEMANTIC_THRESHOLD_DEFAULT)
    ap.add_argument("--semantic_min_units", type=int,   default=SEMANTIC_MIN_UNITS_DEFAULT)

    ap.add_argument("--enable_ocr",      action="store_true", default=ENABLE_OCR_DEFAULT,
                    help="Run Tesseract OCR on scanned PDF pages without a text layer")
    ap.add_argument("--enable_ocr_pptx", action="store_true", default=ENABLE_OCR_PPTX_DEFAULT,
                    help="Run Tesseract on scanned slides in converted PPTX files")
    ap.add_argument("--ocr_lang",               type=str, default=OCR_LANG_DEFAULT)
    ap.add_argument("--ocr_min_chars_per_page", type=int, default=OCR_MIN_CHARS_PER_PAGE_DEFAULT)
    ap.add_argument("--ocr_dpi",                type=int, default=OCR_DPI_DEFAULT)

    ap.add_argument("--full_page_image_ratio", type=float, default=FULL_PAGE_IMAGE_RATIO_DEFAULT,
                    help="Image area / page area threshold to classify a page as a scanned image")
    ap.add_argument("--min_image_px", type=int, default=MIN_IMAGE_PX_DEFAULT,
                    help="Minimum width or height in pixels to save an extracted image")

    ap.add_argument("--tesseract_cmd", type=str, default=TESSERACT_CMD_DEFAULT)
    ap.add_argument("--tessdata_dir",  type=str, default=TESSDATA_DIR_DEFAULT)
    ap.add_argument("--soffice_cmd",   type=str, default=SOFFICE_CMD_DEFAULT)

    # Vision args stored in cfg for ollama_caption_png() at inference time.
    # They have no effect during ingestion.
    ap.add_argument("--vision_model",     type=str, default=VISION_MODEL_DEFAULT)
    ap.add_argument("--vision_timeout_s", type=int, default=VISION_TIMEOUT_S_DEFAULT)

    ap.add_argument("--log_level",   type=str, default=LOG_LEVEL_DEFAULT,
                    help="DEBUG|INFO|WARN|ERROR")
    ap.add_argument("--log_every_s", type=int, default=LOG_EVERY_S_DEFAULT,
                    help="Heartbeat interval in seconds")

    return ap.parse_args()


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data_dir not found or not a directory: {data_dir}")

    cfg = {
        "image_cache_dir":          args.image_cache_dir,
        "chunk_size_tokens":        args.chunk_size_tokens,
        "chunk_overlap_tokens":     args.chunk_overlap_tokens,
        "min_chunk_tokens":         args.min_chunk_tokens,
        "tokenizer_backend":        args.tokenizer_backend,
        "tokenizer_model":          args.tokenizer_model,
        "enable_section_awareness": args.enable_section_awareness,
        "enable_semantic_chunking": args.enable_semantic_chunking,
        "semantic_model":           args.semantic_model,
        "semantic_threshold":       args.semantic_threshold,
        "semantic_min_units":       args.semantic_min_units,
        "enable_ocr":               args.enable_ocr,
        "enable_ocr_pptx":          args.enable_ocr_pptx,
        "ocr_lang":                 args.ocr_lang,
        "ocr_min_chars_per_page":   args.ocr_min_chars_per_page,
        "ocr_dpi":                  args.ocr_dpi,
        "full_page_image_ratio":    args.full_page_image_ratio,
        "min_image_px":             args.min_image_px,
        "tesseract_cmd":            args.tesseract_cmd,
        "tessdata_dir":             args.tessdata_dir,
        "soffice_cmd":              args.soffice_cmd,
        "ollama_base_url":          OLLAMA_BASE_URL_DEFAULT,
        "vision_model":             args.vision_model,
        "vision_timeout_s":         args.vision_timeout_s,
        "log_level":                args.log_level.upper(),
        "log_every_s":              args.log_every_s,
    }

    resources = RuntimeResources(
        token_counter=TokenCounter(cfg),
        semantic_encoder=SemanticEncoder(cfg),
    )

    log(
        f"config  data_dir={data_dir}  out={out_path}\n"
        f"        chunk={cfg['chunk_size_tokens']}tok  "
        f"overlap={cfg['chunk_overlap_tokens']}tok  "
        f"min={cfg['min_chunk_tokens']}tok\n"
        f"        section_awareness={cfg['enable_section_awareness']}  "
        f"semantic_chunking={cfg['enable_semantic_chunking']}  "
        f"semantic_model_loaded={resources.semantic_encoder.available}\n"
        f"        tokenizer={resources.token_counter.resolved_backend}  "
        f"ocr={cfg['enable_ocr']}  "
        f"image_cache={cfg['image_cache_dir']}",
        cfg=cfg,
    )

    files, records = ingest(
        data_dir=data_dir, out_path=out_path, cfg=cfg, resources=resources
    )
    log(f"done  files={files}  records={records}  output={out_path}", cfg=cfg)


if __name__ == "__main__":
    main()