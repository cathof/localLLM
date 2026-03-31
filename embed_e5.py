#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch


# -----------------------------
# .env loader (minimal, no deps) – same as importDocuments.py
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


# Load .env early so env_* helpers see the values
load_dotenv(".env")


# -----------------------------
# ENV helpers
# -----------------------------
def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() != "" else default


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return int(v)


# -----------------------------
# IO helpers
# -----------------------------
def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {e}") from e


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


def choose_device(name: str) -> torch.device:
    n = (name or "auto").lower().strip()

    if n == "cpu":
        return torch.device("cpu")
    if n in {"mps", "metal"}:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if n == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class EmbedConfig:
    input_jsonl: Path
    output_npz: Path
    output_index: Path
    model_name: str
    batch_size: int
    max_length: int
    device: torch.device


# -----------------------------
# Model loading
# -----------------------------
def load_hf_model(model_name: str, device: torch.device):
    """
    Load HF transformer backbone. We implement explicit mean-pooling + normalization,
    which is the recommended approach for E5-style retrieval embeddings.
    """
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: transformers. Install with: pip install transformers") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer


# -----------------------------
# E5 passage embedding
# -----------------------------
def embed_e5_passages(
        texts: List[str],
        *,
        model,
        tokenizer,
        device: torch.device,
        max_length: int,
) -> np.ndarray:
    """
    E5 convention for indexing docs: prefix each chunk with 'passage: '.
    We mean-pool last_hidden_state with attention mask and L2-normalize.
    Returns float32 numpy array [B, D].
    """
    prefixed = ["passage: " + t for t in texts]

    enc = tokenizer(
        prefixed,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, T, D]

        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)                  # [B, D]
        counts = mask.sum(dim=1).clamp(min=1e-9)                  # [B, 1]
        pooled = summed / counts                                  # [B, D]

        pooled = l2_normalize(pooled).to(torch.float32).cpu().numpy()
        return pooled.astype(np.float32)


# -----------------------------
# Pipeline
# -----------------------------
def embed_jsonl(cfg: EmbedConfig) -> Tuple[int, int, int]:
    cfg.output_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_index.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_hf_model(cfg.model_name, cfg.device)

    ids: List[str] = []
    emb_blocks: List[np.ndarray] = []
    n_read = 0
    n_embedded = 0

    batch_texts: List[str] = []
    batch_ids: List[str] = []
    batch_meta: List[Dict[str, Any]] = []

    with cfg.output_index.open("w", encoding="utf-8") as idx_out:

        def flush() -> None:
            nonlocal n_embedded, batch_texts, batch_ids, batch_meta
            if not batch_texts:
                return

            vecs = embed_e5_passages(
                batch_texts,
                model=model,
                tokenizer=tokenizer,
                device=cfg.device,
                max_length=cfg.max_length,
            )

            if vecs.shape[0] != len(batch_ids):
                raise RuntimeError(f"Embedding mismatch: got {vecs.shape[0]} vectors for {len(batch_ids)} texts")

            ids.extend(batch_ids)
            emb_blocks.append(vecs)
            n_embedded += len(batch_ids)

            for _id, meta in zip(batch_ids, batch_meta):
                idx_row = {
                    "id":              _id,
                    "source_path":     meta.get("source_path"),
                    "source_name":     meta.get("source_name"),
                    "ext":             meta.get("ext"),
                    "chunk_index":     meta.get("chunk_index"),
                    "chunk_len":       meta.get("chunk_len"),
                    "file_sha256":     meta.get("file_sha256"),
                    "pdf_ocr_used":    meta.get("pdf_ocr_used"),
                    "pdf_text_reader": meta.get("pdf_text_reader"),   # new: which library extracted text
                    "ocr_lang":        meta.get("ocr_lang"),           # renamed from ocr_lang_used
                    "embedded_images": meta.get("embedded_images", []), # new: image cache paths for inference
                    "mtime_utc":       meta.get("mtime_utc"),
                }
                idx_out.write(json.dumps(idx_row, ensure_ascii=False) + "\n")

            batch_texts, batch_ids, batch_meta = [], [], []

        for obj in iter_jsonl(cfg.input_jsonl):
            n_read += 1
            _id = obj.get("id")
            text = obj.get("text")
            meta = obj.get("metadata", {})

            if not isinstance(_id, str) or not isinstance(text, str) or not isinstance(meta, dict):
                continue
            if not text.strip():
                continue

            batch_ids.append(_id)
            batch_texts.append(text)
            batch_meta.append(meta)

            if len(batch_texts) >= cfg.batch_size:
                flush()

        flush()

    if not emb_blocks:
        raise RuntimeError(f"No embeddings created from input: {cfg.input_jsonl}")

    embeddings = np.vstack(emb_blocks).astype(np.float32)
    ids_arr = np.array(ids, dtype="U64")
    np.savez_compressed(cfg.output_npz, ids=ids_arr, embeddings=embeddings)

    return n_read, n_embedded, embeddings.shape[1]


# -----------------------------
# CLI (defaults from .env via os.environ)
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Embed prepared.jsonl with multilingual-e5 via PyTorch (MPS/CUDA/CPU).")

    ap.add_argument("--in", dest="input_jsonl", type=str, default=env_str("EMBED_INPUT", "prepared.jsonl"))
    ap.add_argument("--out_npz", type=str, default=env_str("EMBED_OUT_NPZ", "embeddings.npz"))
    ap.add_argument("--out_index", type=str, default=env_str("EMBED_OUT_INDEX", "index.jsonl"))

    ap.add_argument("--model", type=str, default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large-instruct"))
    ap.add_argument("--device", type=str, default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--batch_size", type=int, default=env_int("EMBED_BATCH_SIZE", 64))
    ap.add_argument("--max_length", type=int, default=env_int("EMBED_MAX_LENGTH", 512))

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    cfg = EmbedConfig(
        input_jsonl=Path(args.input_jsonl).resolve(),
        output_npz=Path(args.out_npz).resolve(),
        output_index=Path(args.out_index).resolve(),
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    if not cfg.input_jsonl.exists():
        raise SystemExit(f"Input JSONL not found: {cfg.input_jsonl}")

    n_read, n_embedded, dim = embed_jsonl(cfg)

    print(f"OK. Device={device} | Model={cfg.model_name}")
    print(f"Read records: {n_read} | Embedded rows: {n_embedded} | dim={dim}")
    print(f"Embeddings: {cfg.output_npz}")
    print(f"Index: {cfg.output_index}")


if __name__ == "__main__":
    main()