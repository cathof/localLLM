#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple

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

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}


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
    return v if v is not None and v != "" else default


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


# -----------------------------
# Global config (from .env)
# -----------------------------
load_dotenv(".env")

DATA_DIR_DEFAULT = env_str("DATA_DIR", "./data")
OUT_JSONL_DEFAULT = env_str("OUT_JSONL", "prepared.jsonl")

CHUNK_SIZE_DEFAULT = env_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP_DEFAULT = env_int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS_DEFAULT = env_int("MIN_CHUNK_CHARS", 200)

ENABLE_OCR_DEFAULT = env_bool("ENABLE_OCR", False)
OCR_LANG_DEFAULT = env_str("OCR_LANG", "deu")
OCR_MIN_CHARS_PER_PAGE_DEFAULT = env_int("OCR_MIN_CHARS_PER_PAGE", 80)

TESSERACT_CMD_DEFAULT = os.environ.get("TESSERACT_CMD")
TESSDATA_DIR_DEFAULT = os.environ.get("TESSDATA_DIR")
# TESSDATA_PREFIX is a standard env var used by tesseract, we just pass it through
POPPLER_PATH_DEFAULT = os.environ.get("POPPLER_PATH")


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
def sha256_hex(b: bytes) -> str:
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
# File reading
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
    reader = PdfReader(str(path))
    pages = len(reader.pages) or 1
    parts: list[str] = []
    for page in reader.pages:
        t = (page.extract_text() or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip(), pages


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
    # 1) explicit tessdata dir
    if explicit_dir:
        p = Path(explicit_dir)
        if p.exists():
            return str(p)
        raise RuntimeError(f"TESSDATA_DIR set but does not exist: {explicit_dir}")

    # 2) TESSDATA_PREFIX (standard)
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix:
        pref = Path(prefix).expanduser()
        if pref.name == "tessdata" and pref.exists():
            return str(pref)
        cand = pref / "tessdata"
        if cand.exists():
            return str(cand)

    # 3) ask tesseract itself
    try:
        r = subprocess.run([tesseract_cmd, "--print-tessdata-dir"], capture_output=True, text=True)
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
) -> Tuple[str, dict]:
    """
    Returns (text, meta).
    - Try pypdf text extraction
    - If chars/page below threshold and OCR enabled: OCR via pdf2image + pytesseract
    """
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
        meta["pdf_ocr_used"] = False
        meta["pdf_text_chars"] = len(text)
        return text, meta

    if not enable_ocr:
        # OCR disabled: return whatever we got (maybe empty)
        return text, meta

    # OCR enabled and likely scan
    if convert_from_path is None:
        raise RuntimeError("Missing dependency: pdf2image (pip install pdf2image)")
    if pytesseract is None or image_to_string is None:
        raise RuntimeError("Missing dependency: pytesseract (pip install pytesseract)")

    print(f"No/low text found in {path} (chars/page={chars_per_page}), attempting OCR...")
    print(f"[OCR] file={path.name} lang={ocr_lang} dpi=300", flush=True)

    resolved_tesseract = _resolve_tesseract_cmd(tesseract_cmd)
    resolved_tessdata = _resolve_tessdata_dir(resolved_tesseract, tessdata_dir)

    pytesseract.pytesseract.tesseract_cmd = resolved_tesseract

    cfg = ""
    if resolved_tessdata:
        cfg = f"--tessdata-dir {resolved_tessdata}"

    try:
        images = convert_from_path(str(path), poppler_path=poppler_path, dpi=300)
        ocr_parts: list[str] = []
        for img in images:
            ocr_text = (image_to_string(img, lang=ocr_lang, config=cfg) or "").strip()
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
        print(f"[WARN] OCR failed for {path}: {e}", file=sys.stderr)
        # Keep pipeline type-safe
        return "", meta


def read_any(path: Path, *, cfg: dict) -> Tuple[str, dict]:
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
        )
    raise ValueError(f"Unsupported extension: {ext}")


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

    lines = []
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

            if chunk_overlap > 0:
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
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_chars: int,
        cfg: dict,
) -> Tuple[int, int]:
    files_processed = 0
    chunks_written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for path in iter_files(data_dir):
            files_processed += 1  # count every discovered file
            rel_path = str(path.relative_to(data_dir))

            try:
                raw, read_meta = read_any(path, cfg=cfg)
            except Exception as e:
                print(f"[WARN] Failed to read {rel_path}: {e}", file=sys.stderr)
                continue

            text = normalize_text(raw)
            if not text:
                continue

            try:
                fhash = file_sha256(path)
            except Exception as e:
                print(f"[WARN] Failed hashing {rel_path}: {e}", file=sys.stderr)
                fhash = sha256_hex(rel_path.encode("utf-8"))

            base_meta = build_metadata(path, rel_path, text)
            base_meta["file_sha256"] = fhash
            base_meta.update(read_meta)

            for chunk_index, ctext in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
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
                record = ChunkRecord(id=rec_id, text=ctext, metadata=meta)
                out.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                chunks_written += 1

    return files_processed, chunks_written


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
    ap.add_argument("--ocr_lang", type=str, default=OCR_LANG_DEFAULT)
    ap.add_argument("--ocr_min_chars_per_page", type=int, default=OCR_MIN_CHARS_PER_PAGE_DEFAULT)

    ap.add_argument("--poppler_path", type=str, default=POPPLER_PATH_DEFAULT)
    ap.add_argument("--tesseract_cmd", type=str, default=TESSERACT_CMD_DEFAULT)
    ap.add_argument("--tessdata_dir", type=str, default=TESSDATA_DIR_DEFAULT)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data_dir not found or not a directory: {data_dir}")

    cfg = {
        "enable_ocr": args.enable_ocr,
        "ocr_lang": args.ocr_lang,
        "ocr_min_chars_per_page": args.ocr_min_chars_per_page,
        "poppler_path": args.poppler_path,
        "tesseract_cmd": args.tesseract_cmd,
        "tessdata_dir": args.tessdata_dir,
    }

    files_processed, chunks_written = ingest(
        data_dir=data_dir,
        out_path=out_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        cfg=cfg,
    )

    print(f"OK. Files processed: {files_processed} | Chunks written: {chunks_written}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()