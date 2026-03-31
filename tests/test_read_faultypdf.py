from __future__ import annotations
import json

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None  # type: ignore

try:
    import pytesseract
    from pytesseract import image_to_string
except Exception:
    pytesseract = None  # type: ignore
    image_to_string = None  # type: ignore

try:
    import pdfplumber
except Exception:
    pdfplumber = None  # type: ignore


def test_read_faulty_pdf(
        pdf_path: str | Path,
        *,
        enable_ocr: bool = False,
        ocr_lang: str = "deu",
        poppler_path: Optional[str] = None,
        tesseract_cmd: Optional[str] = None,
        tessdata_dir: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Testet ein problematisches PDF mit mehreren Readern.

    Reihenfolge:
    1. pypdf
    2. PyMuPDF
    3. OCR

    Gibt den besten Text plus Debug-Metadaten zurück.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF nicht gefunden: {path}")

    meta: dict = {
        "source_path": str(path),
        "reader_used": None,
        "pages": 0,
        "text_len": 0,
        "pypdf_ok": False,
        "pypdf_text_len": 0,
        "pymupdf_ok": False,
        "pymupdf_text_len": 0,
        "pdfplumber_ok": False,
        "pdfplumber_text_len": 0,
        "ocr_used": False,
        "ocr_text_len": 0,
    }

    print(f"\n[TEST] PDF: {path}")

    # 1) pypdf
    pypdf_text = ""
    pypdf_pages = 0
    if PdfReader is not None:
        try:
            pypdf_text, pypdf_pages = _extract_pdf_text_pypdf(path)
            meta["pypdf_ok"] = True
            meta["pypdf_text_len"] = len(pypdf_text)
            print(f"[TEST] pypdf OK | pages={pypdf_pages} | chars={len(pypdf_text)}")
        except Exception as e:
            print(f"[TEST] pypdf FAILED | {e}")
    else:
        print("[TEST] pypdf nicht verfügbar")

    # 2) PyMuPDF
    pymupdf_text = ""
    pymupdf_pages = 0
    if fitz is not None:
        try:
            pymupdf_text, pymupdf_pages = _extract_pdf_text_pymupdf(path)
            meta["pymupdf_ok"] = True
            meta["pymupdf_text_len"] = len(pymupdf_text)
            print(f"[TEST] PyMuPDF OK | pages={pymupdf_pages} | chars={len(pymupdf_text)}")
        except Exception as e:
            print(f"[TEST] PyMuPDF FAILED | {e}")
    else:
        print("[TEST] PyMuPDF nicht verfügbar")

    # 3) pdfplumber
    pdfplumber_text = ""
    pdfplumber_pages = 0
    if pdfplumber is not None:
        try:
            pdfplumber_text, pdfplumber_pages = _extract_pdf_text_pdfplumber(path)
            meta["pdfplumber_ok"] = True
            meta["pdfplumber_text_len"] = len(pdfplumber_text)
            print(f"[TEST] pdfplumber OK | pages={pdfplumber_pages} | chars={len(pdfplumber_text)}")
        except Exception as e:
            print(f"[TEST] pdfplumber FAILED | {e}")
    else:
        print("[TEST] pdfplumber nicht verfügbar")

    # Beste Textquelle wählen
    candidates = [
        ("pypdf", pypdf_text, pypdf_pages),
        ("pymupdf", pymupdf_text, pymupdf_pages),
        ("pdfplumber", pdfplumber_text, pdfplumber_pages),
    ]
    candidates = [(name, text, pages) for name, text, pages in candidates if text.strip()]

    if candidates:
        best_name, best_text, best_pages = max(candidates, key=lambda item: len(item[1]))
        meta["reader_used"] = best_name
        meta["pages"] = best_pages
        meta["text_len"] = len(best_text)
        print(f"[TEST] Best text source: {best_name} | chars={len(best_text)}")
        return best_text, meta

    # 3) OCR als letzter Fallback
    if enable_ocr:
        try:
            ocr_text, ocr_pages = _extract_pdf_text_ocr(
                path,
                ocr_lang=ocr_lang,
                poppler_path=poppler_path,
                tesseract_cmd=tesseract_cmd,
                tessdata_dir=tessdata_dir,
            )
            meta["reader_used"] = "ocr"
            meta["pages"] = ocr_pages
            meta["text_len"] = len(ocr_text)
            meta["ocr_used"] = True
            meta["ocr_text_len"] = len(ocr_text)
            print(f"[TEST] OCR OK | pages={ocr_pages} | chars={len(ocr_text)}")
            return ocr_text, meta
        except Exception as e:
            print(f"[TEST] OCR FAILED | {e}")

    print("[TEST] Kein brauchbarer Text extrahiert")
    return "", meta


def _extract_pdf_text_pypdf(path: Path) -> Tuple[str, int]:
    if PdfReader is None:
        raise RuntimeError("pypdf nicht installiert")

    reader = PdfReader(str(path))
    pages = len(reader.pages) or 1
    parts: list[str] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            #text = (page.extract_text() or "").strip()
            text = (page.extract_text(extraction_mode="layout") or "").strip()
            if text:
                parts.append(text)
        except Exception as e:
            print(f"[TEST] pypdf page {page_index} failed: {e}")

    return "\n\n".join(parts).strip(), pages


def _extract_pdf_text_pymupdf(path: Path) -> Tuple[str, int]:
    if fitz is None:
        raise RuntimeError("PyMuPDF nicht installiert")
    assert fitz is not None
    doc = fitz.open(str(path))
    pages = len(doc)
    parts: list[str] = []

    for page_index in range(pages):
        try:
            page = doc.load_page(page_index)
            text = (page.get_text("text") or "").strip()
            if text:
                parts.append(text)
        except Exception as e:
            print(f"[TEST] PyMuPDF page {page_index + 1} failed: {e}")

    return "\n\n".join(parts).strip(), pages


def _extract_pdf_text_pdfplumber(path: Path) -> Tuple[str, int]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber nicht installiert")

    parts: list[str] = []
    page_count = 0

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                # 1. Tabellen extrahieren
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    if not table:
                        continue
                    # Zeilen zu Text zusammenfügen
                    rows = []
                    for row in table:
                        # None zu leerem String machen
                        cleaned_row = [str(cell or "").strip() for cell in row]
                        # Nur Zeilen nehmen, die nicht komplett leer sind
                        if any(cleaned_row):
                            rows.append(" | ".join(cleaned_row))
                    if rows:
                        table_texts.append("\n".join(rows))

                # 2. Text extrahieren (ohne Tabellen zu löschen, um Kontext zu behalten)
                page_text = (page.extract_text() or "").strip()

                # Zusammenführen
                if table_texts:
                    combined = page_text + "\n\n[TABLES]\n" + "\n\n".join(table_texts)
                    parts.append(combined)
                elif page_text:
                    parts.append(page_text)

            except Exception as e:
                print(f"[TEST] pdfplumber page {page_index} failed: {e}")

    return "\n\n".join(parts).strip(), page_count


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

    raise RuntimeError("tesseract nicht gefunden")


def _resolve_tessdata_dir(tesseract_cmd: str, explicit_dir: Optional[str]) -> Optional[str]:
    if explicit_dir:
        p = Path(explicit_dir)
        if p.exists():
            return str(p)
        raise RuntimeError(f"TESSDATA_DIR existiert nicht: {explicit_dir}")

    try:
        result = subprocess.run(
            [str(tesseract_cmd), "--print-tessdata-dir"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (result.stdout or "").strip()
        if out and Path(out).exists():
            return out
    except Exception:
        pass

    return None


def _extract_pdf_text_ocr(
        path: Path,
        *,
        ocr_lang: str,
        poppler_path: Optional[str],
        tesseract_cmd: Optional[str],
        tessdata_dir: Optional[str],
) -> Tuple[str, int]:
    if convert_from_path is None:
        raise RuntimeError("pdf2image nicht installiert")
    if pytesseract is None or image_to_string is None:
        raise RuntimeError("pytesseract nicht installiert")

    resolved_tesseract = _resolve_tesseract_cmd(tesseract_cmd)
    resolved_tessdata = _resolve_tessdata_dir(resolved_tesseract, tessdata_dir)

    pytesseract.pytesseract.tesseract_cmd = resolved_tesseract

    tesseract_config = ""
    if resolved_tessdata:
        tesseract_config = f"--tessdata-dir {resolved_tessdata}"

    images = convert_from_path(str(path), poppler_path=poppler_path, dpi=300)
    parts: list[str] = []

    for page_index, img in enumerate(images, start=1):
        try:
            text = (image_to_string(img, lang=ocr_lang, config=tesseract_config) or "").strip()
            if text:
                parts.append(text)
        except Exception as e:
            print(f"[TEST] OCR page {page_index} failed: {e}")

    return "\n\n".join(parts).strip(), len(images)

if __name__ == "__main__":
    text, meta = test_read_faulty_pdf(
        "/Users/catherinehofstetter/Documents/ZHAW/MasterArbeit/LocalLLM/data/VOR-0062.pptx",
        enable_ocr=True,
        ocr_lang="deu",
        poppler_path=None,
        tesseract_cmd=None,
        tessdata_dir=None,
    )

    print("\n--- META ---")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    print("\n--- TEXT PREVIEW ---")
    print(text[:10000])