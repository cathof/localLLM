#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set, Tuple

import numpy as np

import importDocuments_structural as imp
import embed_e5 as emb


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _now_tag() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() else default


def _atomic_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dst))


def _iter_jsonl_ids(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            _id = obj.get("id")
            if isinstance(_id, str) and _id:
                yield _id


def _load_existing_ids(index_jsonl: Path) -> Set[str]:
    if not index_jsonl.exists():
        return set()
    return set(_iter_jsonl_ids(index_jsonl))


# -----------------------------------------------------------------------------
# Core: ingest only one case_id
# -----------------------------------------------------------------------------

def _build_ingest_cfg() -> dict:
    """
    Build the same cfg dict shape that importDocuments_structural.main() uses,
    but seeded from the module defaults (which already read .env).
    """
    return {
        "image_cache_dir": imp.IMAGE_CACHE_DIR_DEFAULT,

        "chunk_size_tokens": imp.CHUNK_SIZE_TOKENS_DEFAULT,
        "chunk_overlap_tokens": imp.CHUNK_OVERLAP_TOKENS_DEFAULT,
        "min_chunk_tokens": imp.MIN_CHUNK_TOKENS_DEFAULT,

        "tokenizer_backend": imp.TOKENIZER_BACKEND_DEFAULT,
        "tokenizer_model": imp.TOKENIZER_MODEL_DEFAULT,

        "enable_section_awareness": imp.ENABLE_SECTION_AWARENESS_DEFAULT,
        "enable_semantic_chunking": imp.ENABLE_SEMANTIC_CHUNKING_DEFAULT,
        "semantic_model": imp.SEMANTIC_MODEL_DEFAULT,
        "semantic_threshold": imp.SEMANTIC_THRESHOLD_DEFAULT,
        "semantic_min_units": imp.SEMANTIC_MIN_UNITS_DEFAULT,

        "enable_ocr": imp.ENABLE_OCR_DEFAULT,
        "enable_ocr_pptx": imp.ENABLE_OCR_PPTX_DEFAULT,
        "ocr_lang": imp.OCR_LANG_DEFAULT,
        "ocr_min_chars_per_page": imp.OCR_MIN_CHARS_PER_PAGE_DEFAULT,
        "ocr_dpi": imp.OCR_DPI_DEFAULT,

        "full_page_image_ratio": imp.FULL_PAGE_IMAGE_RATIO_DEFAULT,
        "min_image_px": imp.MIN_IMAGE_PX_DEFAULT,
        "vision_page_render_dpi": imp.VISION_PAGE_RENDER_DPI_DEFAULT,
        "pdf_text_extractor": imp.PDF_TEXT_EXTRACTOR_DEFAULT,
        "pdf_suspicious_run_len": imp.PDF_SUSPICIOUS_RUN_LEN_DEFAULT,

        "tesseract_cmd": imp.TESSERACT_CMD_DEFAULT,
        "tessdata_dir": imp.TESSDATA_DIR_DEFAULT,
        "soffice_cmd": imp.SOFFICE_CMD_DEFAULT,

        "ollama_base_url": imp.OLLAMA_BASE_URL_DEFAULT,
        "vision_model": imp.VISION_MODEL_DEFAULT,
        "vision_timeout_s": imp.VISION_TIMEOUT_S_DEFAULT,

        "log_level": imp.LOG_LEVEL_DEFAULT,
        "log_every_s": imp.LOG_EVERY_S_DEFAULT,
    }


def ingest_case_to_jsonl(
        *,
        case_id: str,
        cases_dir: Path,
        out_jsonl: Path,
        existing_ids: Set[str],
        cfg: dict,
        resources: imp.RuntimeResources,
) -> Tuple[int, int]:
    """
    Writes ONLY new chunk records for the given case_id to out_jsonl.
    Returns (files_processed, records_written).
    """
    st = imp.RunStats(t0=imp.time.time(), last_heartbeat=imp.time.time())
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    records_written = 0
    seen_new: Set[str] = set()

    with out_jsonl.open("w", encoding="utf-8") as out:
        for path, source_ctx in imp.iter_source_files(None, cases_dir):
            if source_ctx.source_kind != "case_material":
                continue
            if source_ctx.case_id != case_id:
                continue

            files_processed += 1
            st.files = files_processed

            rel = source_ctx.rel_path
            imp.log(
                f"append-case ingest start case_id={case_id} file={rel}",
                cfg=cfg,
            )

            try:
                wrote_for_file = 0
                for rec in imp.iter_records_for_path(
                        path,
                        source_ctx=source_ctx,
                        cfg=cfg,
                        st=st,
                        resources=resources,
                ):
                    # Dedup vs existing store + within this run
                    if rec.id in existing_ids or rec.id in seen_new:
                        continue

                    out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    records_written += 1
                    wrote_for_file += 1
                    st.records = records_written
                    seen_new.add(rec.id)

                    imp.heartbeat(st, cfg=cfg, extra=f"current={rel}")

                imp.log(
                    f"append-case ingest done file={rel} new_records={wrote_for_file}",
                    cfg=cfg,
                )

            except Exception as e:
                imp.log(
                    f"append-case ingest failed file={rel}: {e}",
                    level="WARN",
                    cfg=cfg,
                    err=True,
                )

    return files_processed, records_written


# -----------------------------------------------------------------------------
# Embedding + merge
# -----------------------------------------------------------------------------

def embed_delta(
        *,
        delta_prepared: Path,
        delta_npz: Path,
        delta_index: Path,
) -> Tuple[int, int, int]:
    """
    Embed delta_prepared into delta_npz + delta_index using embed_e5.py config defaults (.env).
    Returns (n_read, n_embedded, dim).
    """
    model = emb.env_str("EMBED_MODEL", "intfloat/multilingual-e5-large")
    device_name = emb.env_str("EMBED_DEVICE", "auto")
    batch_size = emb.env_int("EMBED_BATCH_SIZE", 64)
    max_length = emb.env_int("EMBED_MAX_LENGTH", 512)

    device = emb.choose_device(device_name)

    cfg = emb.EmbedConfig(
        input_jsonl=delta_prepared.resolve(),
        output_npz=delta_npz.resolve(),
        output_index=delta_index.resolve(),
        model_name=model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    return emb.embed_jsonl(cfg)


def merge_append_jsonl(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        _atomic_replace(src, dst)
        return
    with dst.open("a", encoding="utf-8") as out, src.open("r", encoding="utf-8") as inp:
        for line in inp:
            if line.strip():
                out.write(line.rstrip("\n") + "\n")


def merge_append_npz(dst_npz: Path, delta_npz: Path) -> int:
    """
    Append delta embeddings into dst_npz. Returns number of appended rows.
    """
    dst_npz.parent.mkdir(parents=True, exist_ok=True)

    if not dst_npz.exists():
        _atomic_replace(delta_npz, dst_npz)
        with np.load(dst_npz) as z:
            return int(z["embeddings"].shape[0])

    with np.load(dst_npz) as z0:
        ids0 = z0["ids"]
        emb0 = z0["embeddings"].astype(np.float32)

    with np.load(delta_npz) as z1:
        ids1 = z1["ids"]
        emb1 = z1["embeddings"].astype(np.float32)

    if emb0.shape[1] != emb1.shape[1]:
        raise RuntimeError(f"Embedding dim mismatch: {emb0.shape[1]} vs {emb1.shape[1]}")

    # Safety dedup (should already be filtered during ingest)
    existing = set(map(str, ids0.tolist()))
    keep_idx = [i for i, _id in enumerate(ids1.tolist()) if str(_id) not in existing]
    if not keep_idx:
        return 0

    ids1k = ids1[keep_idx]
    emb1k = emb1[keep_idx, :]

    merged_ids = np.concatenate([ids0, ids1k]).astype("U64")
    merged_emb = np.vstack([emb0, emb1k]).astype(np.float32)

    tmp = dst_npz.with_suffix(dst_npz.suffix + ".tmp")
    np.savez_compressed(tmp, ids=merged_ids, embeddings=merged_emb)
    _atomic_replace(tmp, dst_npz)
    return int(emb1k.shape[0])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # Ensure .env is loaded even if this script is used standalone.
    imp.load_dotenv(".env")
    emb.load_dotenv(".env")

    ap = argparse.ArgumentParser(
        description="Append a single case (case_id) to the case-material RAG store (chunk + embed + merge)."
    )
    ap.add_argument("--case_id", required=True, help="e.g. case_12")

    # Defaults: from .env if set, otherwise the usual convention
    ap.add_argument("--cases_dir", default=imp.CASES_DIR_DEFAULT or "./cases")

    ap.add_argument("--out_jsonl", default=_env_str("OUT_JSONL2", "artefacts/prepared_materials.jsonl"))
    ap.add_argument("--out_npz", default=_env_str("EMBED_OUT_NPZ2", "artefacts/embeddings_materials.npz"))
    ap.add_argument("--out_index", default=_env_str("EMBED_OUT_INDEX2", "artefacts/index_materials.jsonl"))

    ap.add_argument("--delta_dir", default="./.delta_ingest", help="where delta artifacts are written")
    ap.add_argument("--keep_delta", action="store_true", help="do not delete delta artifacts")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    case_id = str(args.case_id).strip()
    cases_dir = Path(args.cases_dir).resolve()

    out_jsonl = Path(args.out_jsonl).resolve()
    out_npz = Path(args.out_npz).resolve()
    out_index = Path(args.out_index).resolve()

    delta_dir = Path(args.delta_dir).resolve()
    delta_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{case_id}__{_now_tag()}"
    delta_prepared = delta_dir / f"prepared__{tag}.jsonl"
    delta_npz = delta_dir / f"embeddings__{tag}.npz"
    delta_index = delta_dir / f"index__{tag}.jsonl"

    if not cases_dir.exists():
        raise SystemExit(f"cases_dir not found: {cases_dir}")

    # Existing IDs for fast "only embed what is new"
    existing_ids = _load_existing_ids(out_index)

    # Build ingestion resources from module defaults (already .env-driven)
    cfg = _build_ingest_cfg()
    resources = imp.RuntimeResources(
        token_counter=imp.TokenCounter(cfg),
        semantic_encoder=imp.SemanticEncoder(cfg),
    )

    imp.log(
        f"append-case start case_id={case_id} cases_dir={cases_dir}",
        cfg=cfg,
    )
    imp.log(
        f"store out_jsonl={out_jsonl.name} out_npz={out_npz.name} out_index={out_index.name}",
        cfg=cfg,
    )

    files_processed, new_records = ingest_case_to_jsonl(
        case_id=case_id,
        cases_dir=cases_dir,
        out_jsonl=delta_prepared,
        existing_ids=existing_ids,
        cfg=cfg,
        resources=resources,
    )

    if new_records == 0:
        imp.log(f"append-case nothing to do (no new chunks) case_id={case_id}", cfg=cfg)
        if not args.keep_delta:
            for p in (delta_prepared, delta_npz, delta_index):
                if p.exists():
                    p.unlink()
        return 0

    n_read, n_embedded, dim = embed_delta(
        delta_prepared=delta_prepared,
        delta_npz=delta_npz,
        delta_index=delta_index,
    )

    # Merge into main store
    merge_append_jsonl(out_jsonl, delta_prepared)
    merge_append_jsonl(out_index, delta_index)
    appended = merge_append_npz(out_npz, delta_npz)

    imp.log(
        f"append-case OK case_id={case_id} files={files_processed} "
        f"new_chunks={new_records} embedded={n_embedded} appended={appended} dim={dim}",
        cfg=cfg,
    )

    if not args.keep_delta:
        for p in (delta_prepared, delta_npz, delta_index):
            if p.exists():
                p.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())