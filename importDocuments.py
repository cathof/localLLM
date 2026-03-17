#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple

# Reduce logs
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Optional deps
try:
    from pdf2image import convert_from_path  # pip install pdf2image
except Exception:
    convert_from_path = None  # type: ignore

try:
    import pytesseract  # pip install pytesseract
    from pytesseract import image_to_string
except Exception:
    pytesseract = None  # type: ignore
    image_to_string = None  # type: ignore

try:
    from docx import Document  # pip install python-docx
except Exception:
    Document = None  # type: ignore

try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None  # type: ignore


SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".pptx"}


# -----------------------------
# .env loader (minimal, no deps)
# -----------------------------
def load_dotenv(dotenv_path: str | Path = ".env") -> None:
    """
    Load KEY=VALUE pairs from .env into process environment.
    Does not overwrite existing environment variables.
    """
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


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# Load .env early
load_dotenv(".env")

# -----------------------------
# Defaults (from .env)
# -----------------------------
DATA_DIR_DEFAULT = env_str("DATA_DIR", "./data")
OUT_JSONL_DEFAULT = env_str("OUT_JSONL", "prepared.jsonl")

CHUNK_SIZE_DEFAULT = env_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP_DEFAULT = env_int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS_DEFAULT = env_int("MIN_CHUNK_CHARS", 200)

ENABLE_OCR_DEFAULT = env_bool("ENABLE_OCR", False)
ENABLE_OCR_PPTX_DEFAULT = env_bool("ENABLE_OCR_PPTX", False)
OCR_LANG_DEFAULT = env_str("OCR_LANG", "deu")
OCR_MIN_CHARS_PER_PAGE_DEFAULT = env_int("OCR_MIN_CHARS_PER_PAGE", 80)

TESSERACT_CMD_DEFAULT = os.environ.get("TESSERACT_CMD")
TESSDATA_DIR_DEFAULT = os.environ.get("TESSDATA_DIR")
POPPLER_PATH_DEFAULT = os.environ.get("POPPLER_PATH")
SOFFICE_CMD_DEFAULT = os.environ.get("SOFFICE_CMD")

ENABLE_VISION_CAPTIONS_DEFAULT = env_bool("ENABLE_VISION_CAPTIONS", False)
VISION_BACKEND_DEFAULT = env_str("VISION_BACKEND", "ollama")
VISION_MODEL_DEFAULT = env_str("VISION_MODEL", "qwen2.5vl:latest")
VISION_TIMEOUT_S_DEFAULT = env_int("VISION_TIMEOUT_S", 180)
VISION_PROMPT_DEFAULT = env_str(
    "VISION_PROMPT",
    (
        "Beschreibe die Folie präzise:\n"
        "- Titel (falls erkennbar)\n"
        "- Kernaussagen als Bulletpoints (max. 8)\n"
        "- sichtbare Labels/Legenden/Tabellen kurz (falls vorhanden)\n"
        "- falls nur Foto ohne Text: Motiv und Kontext knapp\n"
        "Keine Halluzinationen, nur was sichtbar ist."
    ),
)
VISION_DPI_DEFAULT = env_int("VISION_DPI", 150)
VISION_SKIP_SHORT_DEFAULT = env_int("VISION_SKIP_SHORT", 60)

OLLAMA_BASE_URL_DEFAULT = env_str("OLLAMA_BASE_URL", "http://localhost:11434")

LOG_LEVEL_DEFAULT = env_str("LOG_LEVEL", "INFO").upper()
LOG_EVERY_S_DEFAULT = env_int("LOG_EVERY_S", 10)
VISION_LOG_EVERY_N_DEFAULT = env_int("VISION_LOG_EVERY_N", 5)


# -----------------------------
# Tiny, elegant logging
# -----------------------------
def _lvl(x: str) -> int:
    return {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}.get(x, 20)


def log(msg: str, *, level: str = "INFO", cfg: Optional[dict] = None, err: bool = False) -> None:
    cfg = cfg or {}
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
    files: int = 0
    records: int = 0


def heartbeat(st: RunStats, *, cfg: dict, extra: str = "") -> None:
    every_s = int(cfg.get("log_every_s") or LOG_EVERY_S_DEFAULT)
    now = time.time()
    if now - st.last_heartbeat >= every_s:
        dt = now - st.t0
        tail = f" {extra}".rstrip()
        log(f"heartbeat t={dt:.0f}s files={st.files} records={st.records}{tail}", level="INFO", cfg=cfg)
        st.last_heartbeat = now


# -----------------------------
# Models
# -----------------------------
@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict


# -----------------------------
# Helpers: hashing, metadata
# -----------------------------
def sha256_hex(b: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_chunk_id(file_hash: str, rel_path: str, chunk_index: int, chunk_text: str) -> str:
    payload = f"{file_hash}|{rel_path}|{chunk_index}|{sha256_hex(chunk_text.encode('utf-8', errors='replace'))}"
    return sha256_hex(payload.encode("utf-8"))


def build_metadata(path: Path, rel_path: str, text: str) -> dict:
    st = path.stat()
    return {
        "source_path": rel_path.replace("\\", "/"),
        "source_name": path.name,
        "ext": path.suffix.lower(),
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "ingested_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "text_len": len(text),
    }


# -----------------------------
# Normalization
# -----------------------------
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTINEW_RE = re.compile(r"\n{3,}")
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


# -----------------------------
# Chunking
# -----------------------------
def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[Tuple[int, str]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paras = split_into_paragraphs(text)
    if not paras:
        return

    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    def flush() -> Optional[str]:
        nonlocal current, current_len
        if not current:
            return None
        out = "\n\n".join(current).strip()
        current = []
        current_len = 0
        return out

    for para in paras:
        if len(para) > chunk_size * 2:
            out = flush()
            if out:
                yield chunk_idx, out
                chunk_idx += 1

            start = 0
            while start < len(para):
                end = min(len(para), start + chunk_size)
                piece = para[start:end].strip()
                if piece:
                    yield chunk_idx, piece
                    chunk_idx += 1
                start = max(end - chunk_overlap, end)
            continue

        if current_len and (current_len + 2 + len(para)) > chunk_size:
            out = flush()
            if out:
                yield chunk_idx, out
                chunk_idx += 1

            if chunk_overlap > 0 and out:
                tail = out[-chunk_overlap:].strip()
                if tail:
                    current = [tail]
                    current_len = len(tail)

        current.append(para)
        current_len += (2 if current_len else 0) + len(para)

    out = flush()
    if out:
        yield chunk_idx, out


# -----------------------------
# File reading: TXT, DOCX, PDF (+ optional OCR)
# -----------------------------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def read_docx(path: Path) -> str:
    if Document is None:
        raise RuntimeError("Missing dependency: python-docx (pip install python-docx)")
    doc = Document(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def _extract_pdf_text(path: Path) -> Tuple[str, int]:
    if PdfReader is None:
        raise RuntimeError("Missing dependency: pypdf (pip install pypdf)")

    try:
        reader = PdfReader(str(path), strict=False)
        pages = len(reader.pages) or 1
        parts: list[str] = []

        for page in reader.pages:
            try:
                t = (page.extract_text() or "").strip()
            except Exception:
                t = ""
            if t:
                parts.append(t)

        return "\n\n".join(parts).strip(), pages

    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed for {path.name}: {e}") from e


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

    raise RuntimeError("tesseract not found in PATH and TESSERACT_CMD not set.")


def _resolve_tessdata_dir(tesseract_cmd: str, explicit_dir: Optional[str]) -> Optional[str]:
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
            capture_output=True,
            text=True,
            check=False,
        )
        out = (r.stdout or "").strip()
        if out and Path(out).exists():
            return out
    except Exception:
        pass

    return None


def read_pdf(
        file_path: str | Path,
        *,
        enable_ocr: bool,
        ocr_lang: str,
        ocr_min_chars_per_page: int,
        poppler_path: Optional[str],
        tesseract_cmd: Optional[str],
        tessdata_dir: Optional[str],
        cfg_for_log: Optional[dict] = None,
) -> Tuple[str, dict]:
    meta = {
        "pdf_ocr_used": False,
        "pdf_text_chars": 0,
        "pdf_pages": 0,
        "ocr_lang": ocr_lang,
    }

    path = Path(file_path)
    text, pages = _extract_pdf_text(path)
    meta["pdf_pages"] = pages
    meta["pdf_text_chars"] = len(text)

    chars_per_page = meta["pdf_text_chars"] // max(pages, 1)
    needs_ocr = chars_per_page < ocr_min_chars_per_page

    if text and (not needs_ocr or not enable_ocr):
        return text, meta

    if not enable_ocr:
        return text, meta

    if convert_from_path is None:
        raise RuntimeError("Missing dependency: pdf2image (pip install pdf2image)")
    if pytesseract is None or image_to_string is None:
        raise RuntimeError("Missing dependency: pytesseract (pip install pytesseract)")

    log(
        f"OCR start file={path.name} chars/page={chars_per_page} lang={ocr_lang} dpi=300",
        level="INFO",
        cfg=cfg_for_log or {},
    )

    resolved_tesseract = _resolve_tesseract_cmd(tesseract_cmd)
    resolved_tessdata = _resolve_tessdata_dir(resolved_tesseract, tessdata_dir)
    pytesseract.pytesseract.tesseract_cmd = resolved_tesseract

    tcfg = ""
    if resolved_tessdata:
        tcfg = f"--tessdata-dir {resolved_tessdata}"

    try:
        images = convert_from_path(str(path), poppler_path=poppler_path, dpi=300)
        ocr_parts: list[str] = []

        for img in images:
            ocr_text = (image_to_string(img, lang=ocr_lang, config=tcfg) or "").strip()
            if ocr_text:
                ocr_parts.append(ocr_text)

        ocr_text_all = "\n\n".join(ocr_parts).strip()
        meta["pdf_ocr_used"] = True
        meta["pdf_text_chars"] = len(ocr_text_all)
        meta["ocr_lang_used"] = ocr_lang
        if resolved_tessdata:
            meta["tessdata_dir"] = resolved_tessdata
        meta["tesseract_cmd"] = resolved_tesseract
        return ocr_text_all, meta

    except Exception as e:
        log(f"OCR failed file={path.name}: {e}", level="WARN", cfg=cfg_for_log or {}, err=True)
        return "", meta


# -----------------------------
# PPTX -> PDF conversion (LibreOffice/soffice)
# -----------------------------
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
        candidates = [
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
            Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
            ]
        for c in candidates:
            if c.exists():
                return str(c)

    raise RuntimeError(
        "LibreOffice (soffice) not found. Install LibreOffice and ensure 'soffice' is in PATH, "
        "or set SOFFICE_CMD."
    )


def pptx_to_pdf(path: Path, soffice_cmd: Optional[str], *, cfg_for_log: dict) -> Path:
    cmd = resolve_soffice_cmd(soffice_cmd)
    out_dir = path.parent / ".converted"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / f"{path.stem}.pdf"
    if out_pdf.exists() and out_pdf.stat().st_mtime >= path.stat().st_mtime:
        return out_pdf

    log(f"PPTX->PDF convert start file={path.name}", level="INFO", cfg=cfg_for_log)

    subprocess.run(
        [
            str(cmd),
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    if not out_pdf.exists():
        raise RuntimeError(f"PPTX->PDF failed: {out_pdf}")

    return out_pdf


# -----------------------------
# read_any (default text path)
# -----------------------------
def read_any(path: Path, *, rel_path: str, cfg: dict) -> Tuple[str, dict]:
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return read_text_file(path), {}

    if ext == ".docx":
        return read_docx(path), {}

    if ext == ".pdf":
        return read_pdf(
            path,
            enable_ocr=cfg["enable_ocr"],
            ocr_lang=cfg["ocr_lang"],
            ocr_min_chars_per_page=cfg["ocr_min_chars_per_page"],
            poppler_path=cfg["poppler_path"],
            tesseract_cmd=cfg["tesseract_cmd"],
            tessdata_dir=cfg["tessdata_dir"],
            cfg_for_log=cfg,
        )

    if ext == ".pptx":
        pdf_path = pptx_to_pdf(path, cfg.get("soffice_cmd"), cfg_for_log=cfg)
        enable_ocr_pptx = bool(cfg.get("enable_ocr_pptx", False))

        pdf_text, pdf_meta = read_pdf(
            pdf_path,
            enable_ocr=enable_ocr_pptx,
            ocr_lang=cfg["ocr_lang"],
            ocr_min_chars_per_page=cfg["ocr_min_chars_per_page"],
            poppler_path=cfg["poppler_path"],
            tesseract_cmd=cfg["tesseract_cmd"],
            tessdata_dir=cfg["tessdata_dir"],
            cfg_for_log=cfg,
        )

        pdf_meta.update(
            {
                "origin_source_path": rel_path.replace("\\", "/"),
                "origin_source_name": path.name,
                "origin_ext": ".pptx",
                "converted_pdf": str(pdf_path.name),
            }
        )
        return pdf_text, pdf_meta

    raise ValueError(f"Unsupported extension: {ext}")


# -----------------------------
# Vision captions (Ollama)
# -----------------------------
def ollama_caption_png(*, cfg: dict, png_bytes: bytes) -> str:
    try:
        import requests  # pip install requests
    except Exception as e:
        raise RuntimeError("Missing dependency: requests (pip install requests)") from e

    base_url = str(cfg.get("ollama_base_url") or OLLAMA_BASE_URL_DEFAULT).rstrip("/")
    model = str(cfg.get("vision_model") or "").strip()
    if not model:
        raise RuntimeError("VISION_MODEL is required for vision captions (e.g. qwen2.5vl:7b)")

    prompt = str(cfg.get("vision_prompt") or "").strip() or VISION_PROMPT_DEFAULT
    timeout_s = int(cfg.get("vision_timeout_s") or VISION_TIMEOUT_S_DEFAULT)

    b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Du bist ein präziser Assistent für Folienanalyse."},
            {"role": "user", "content": prompt, "images": [b64]},
        ],
    }

    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    return content.strip() if isinstance(content, str) else ""


def render_pdf_pages_to_png_bytes(
        pdf_path: Path,
        *,
        poppler_path: Optional[str],
        dpi: int,
) -> Iterator[tuple[int, bytes]]:
    if convert_from_path is None:
        raise RuntimeError("Missing dependency: pdf2image (pip install pdf2image)")

    import io

    images = convert_from_path(str(pdf_path), poppler_path=poppler_path, dpi=dpi)
    for i, img in enumerate(images, start=1):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        yield i, buf.getvalue()


# -----------------------------
# Record generation (default + PPTX vision)
# -----------------------------
def iter_records_for_path(path: Path, *, data_dir: Path, cfg: dict, st: RunStats) -> Iterator[ChunkRecord]:
    rel_path = str(path.relative_to(data_dir)).replace("\\", "/")
    ext = path.suffix.lower()

    enable_vision = bool(cfg.get("enable_vision_captions", False))
    vision_backend = str(cfg.get("vision_backend") or "ollama").lower().strip()

    # Branch: PPTX vision captions (1 record per slide)
    if ext == ".pptx" and enable_vision:
        if vision_backend != "ollama":
            raise RuntimeError(f"Unsupported vision backend: {vision_backend}")

        pdf_path = pptx_to_pdf(path, cfg.get("soffice_cmd"), cfg_for_log=cfg)
        fhash = file_sha256(path)

        base_meta = build_metadata(path, rel_path, text="")
        base_meta["file_sha256"] = fhash
        base_meta.update(
            {
                "origin_source_path": rel_path,
                "origin_source_name": path.name,
                "origin_ext": ".pptx",
                "converted_pdf": pdf_path.name,
                "vision_backend": vision_backend,
                "vision_model": str(cfg.get("vision_model") or ""),
            }
        )

        dpi = int(cfg.get("vision_dpi") or VISION_DPI_DEFAULT)
        min_caption_len = int(cfg.get("vision_skip_short") or VISION_SKIP_SHORT_DEFAULT)
        every_n = int(cfg.get("vision_log_every_n") or VISION_LOG_EVERY_N_DEFAULT)

        log(f"PPTX vision start file={rel_path} dpi={dpi} model={cfg.get('vision_model')}", level="INFO", cfg=cfg)

        for slide_index, png_bytes in render_pdf_pages_to_png_bytes(
                pdf_path,
                poppler_path=cfg.get("poppler_path"),
                dpi=dpi,
        ):
            heartbeat(st, cfg=cfg, extra=f"current={rel_path} slide={slide_index}")

            if every_n > 0 and slide_index % every_n == 0:
                log(f"VLM captioning file={rel_path} slide={slide_index}", level="INFO", cfg=cfg)

            t0 = time.time()
            caption = ollama_caption_png(cfg=cfg, png_bytes=png_bytes)
            dt = time.time() - t0

            caption = normalize_text(caption)
            if not caption or len(caption) < min_caption_len:
                if every_n > 0 and slide_index % every_n == 0:
                    log(
                        f"VLM skip file={rel_path} slide={slide_index} len={len(caption)} dt={dt:.1f}s",
                        level="DEBUG",
                        cfg=cfg,
                    )
                continue

            if every_n > 0 and slide_index % every_n == 0:
                log(
                    f"VLM ok file={rel_path} slide={slide_index} len={len(caption)} dt={dt:.1f}s",
                    level="INFO",
                    cfg=cfg,
                )

            text = f"FOLIE {slide_index}\n{caption}".strip()

            meta = dict(base_meta)
            meta.update(
                {
                    "slide_index": slide_index,
                    "image_caption": caption,
                    "text_len": len(text),
                    "chunk_index": 0,
                    "chunk_len": len(text),
                    "chunk_size_chars": 0,
                    "chunk_overlap_chars": 0,
                }
            )

            rec_id = stable_chunk_id(fhash, rel_path, slide_index, text)
            yield ChunkRecord(id=rec_id, text=text, metadata=meta)

        return

    # Branch: default text -> normalize -> chunk
    raw, read_meta = read_any(path, rel_path=rel_path, cfg=cfg)
    text = normalize_text(raw)
    if not text:
        return

    fhash = file_sha256(path)

    base_meta = build_metadata(path, rel_path, text)
    base_meta["file_sha256"] = fhash
    base_meta.update(read_meta)

    chunk_size = int(cfg.get("chunk_size") or CHUNK_SIZE_DEFAULT)
    chunk_overlap = int(cfg.get("chunk_overlap") or CHUNK_OVERLAP_DEFAULT)
    min_chunk_chars = int(cfg.get("min_chunk_chars") or MIN_CHUNK_CHARS_DEFAULT)

    for chunk_index, ctext in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        heartbeat(st, cfg=cfg, extra=f"current={rel_path} chunk={chunk_index}")
        ctext = ctext.strip()
        if len(ctext) < min_chunk_chars:
            continue

        rec_id = stable_chunk_id(fhash, base_meta["source_path"], chunk_index, ctext)

        meta = dict(base_meta)
        meta.update(
            {
                "chunk_index": chunk_index,
                "chunk_len": len(ctext),
                "chunk_size_chars": chunk_size,
                "chunk_overlap_chars": chunk_overlap,
            }
        )

        yield ChunkRecord(id=rec_id, text=ctext, metadata=meta)


# -----------------------------
# Ingestion
# -----------------------------
def iter_files(data_dir: Path) -> Iterator[Path]:
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def ingest(
        data_dir: Path,
        out_path: Path,
        *,
        cfg: dict,
) -> Tuple[int, int]:
    st = RunStats(t0=time.time(), last_heartbeat=time.time())
    files_processed = 0
    records_written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for path in iter_files(data_dir):
            files_processed += 1
            st.files = files_processed

            rel = str(path.relative_to(data_dir)).replace("\\", "/")
            log(f"ingest start file={rel}", level="INFO", cfg=cfg)

            try:
                wrote_this_file = 0
                for rec in iter_records_for_path(path, data_dir=data_dir, cfg=cfg, st=st):
                    out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    records_written += 1
                    wrote_this_file += 1
                    st.records = records_written
                    heartbeat(st, cfg=cfg, extra=f"current={rel}")

                log(f"ingest done file={rel} records={wrote_this_file}", level="INFO", cfg=cfg)

            except Exception as e:
                log(f"ingest failed file={rel}: {e}", level="WARN", cfg=cfg, err=True)

    return files_processed, records_written


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest ./data/** into normalized, chunked JSONL.")

    ap.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT)
    ap.add_argument("--out", type=str, default=OUT_JSONL_DEFAULT)

    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE_DEFAULT)
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP_DEFAULT)
    ap.add_argument("--min_chunk_chars", type=int, default=MIN_CHUNK_CHARS_DEFAULT)

    ap.add_argument("--enable_ocr", action="store_true", default=ENABLE_OCR_DEFAULT)
    ap.add_argument("--enable_ocr_pptx", action="store_true", default=ENABLE_OCR_PPTX_DEFAULT)
    ap.add_argument("--ocr_lang", type=str, default=OCR_LANG_DEFAULT)
    ap.add_argument("--ocr_min_chars_per_page", type=int, default=OCR_MIN_CHARS_PER_PAGE_DEFAULT)

    ap.add_argument("--poppler_path", type=str, default=POPPLER_PATH_DEFAULT)
    ap.add_argument("--tesseract_cmd", type=str, default=TESSERACT_CMD_DEFAULT)
    ap.add_argument("--tessdata_dir", type=str, default=TESSDATA_DIR_DEFAULT)
    ap.add_argument("--soffice_cmd", type=str, default=SOFFICE_CMD_DEFAULT)

    ap.add_argument("--enable_vision_captions", action="store_true", default=ENABLE_VISION_CAPTIONS_DEFAULT)
    ap.add_argument("--vision_backend", type=str, default=VISION_BACKEND_DEFAULT)
    ap.add_argument("--vision_model", type=str, default=VISION_MODEL_DEFAULT)
    ap.add_argument("--vision_timeout_s", type=int, default=VISION_TIMEOUT_S_DEFAULT)
    ap.add_argument("--vision_prompt", type=str, default=VISION_PROMPT_DEFAULT)
    ap.add_argument("--vision_dpi", type=int, default=VISION_DPI_DEFAULT)
    ap.add_argument("--vision_skip_short", type=int, default=VISION_SKIP_SHORT_DEFAULT)

    ap.add_argument("--log_level", type=str, default=LOG_LEVEL_DEFAULT, help="DEBUG|INFO|WARN|ERROR")
    ap.add_argument("--log_every_s", type=int, default=LOG_EVERY_S_DEFAULT, help="Heartbeat interval in seconds")
    ap.add_argument("--vision_log_every_n", type=int, default=VISION_LOG_EVERY_N_DEFAULT, help="Log every N slides")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data_dir not found or not a directory: {data_dir}")

    cfg = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "min_chunk_chars": args.min_chunk_chars,
        "enable_ocr": args.enable_ocr,
        "enable_ocr_pptx": args.enable_ocr_pptx,
        "ocr_lang": args.ocr_lang,
        "ocr_min_chars_per_page": args.ocr_min_chars_per_page,
        "poppler_path": args.poppler_path,
        "tesseract_cmd": args.tesseract_cmd,
        "tessdata_dir": args.tessdata_dir,
        "soffice_cmd": args.soffice_cmd,
        "enable_vision_captions": args.enable_vision_captions,
        "vision_backend": args.vision_backend,
        "vision_model": args.vision_model,
        "vision_timeout_s": args.vision_timeout_s,
        "vision_prompt": args.vision_prompt,
        "vision_dpi": args.vision_dpi,
        "vision_skip_short": args.vision_skip_short,
        "ollama_base_url": OLLAMA_BASE_URL_DEFAULT,
        "log_level": args.log_level.upper(),
        "log_every_s": args.log_every_s,
        "vision_log_every_n": args.vision_log_every_n,
    }

    log(
        f"config data_dir={data_dir} out={out_path} vision={cfg['enable_vision_captions']} "
        f"model={cfg.get('vision_model')} base={cfg.get('ollama_base_url')}",
        level="INFO",
        cfg=cfg,
    )

    files_processed, records_written = ingest(
        data_dir=data_dir,
        out_path=out_path,
        cfg=cfg,
    )

    log(f"done files={files_processed} records={records_written} output={out_path}", level="INFO", cfg=cfg)


if __name__ == "__main__":
    main()