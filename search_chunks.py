#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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


load_dotenv(".env")


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() != "" else default


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return int(v)


# -----------------------------
# Device + math
# -----------------------------
def choose_device(name: str) -> torch.device:
    n = (name or "auto").lower().strip()
    if n == "cpu":
        return torch.device("cpu")
    if n in {"mps", "metal"}:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if n == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


# -----------------------------
# Loading artifacts
# -----------------------------
def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    ids = data["ids"]
    emb = data["embeddings"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings 2D, got {emb.shape}")
    if len(ids) != emb.shape[0]:
        raise ValueError(f"ids length mismatch: {len(ids)} vs {emb.shape[0]}")
    return ids, emb


def load_index(index_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in {index_path} at line {line_no}: {e}") from e
    return rows


def load_prepared_text_map(prepared_jsonl: Path) -> Dict[str, str]:
    """
    Loads id -> text from prepared.jsonl.
    This can be memory-heavy for very large corpora, but is fine for ~10k chunks.
    """
    m: Dict[str, str] = {}
    with prepared_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in {prepared_jsonl} at line {line_no}: {e}") from e
            _id = o.get("id")
            txt = o.get("text")
            if isinstance(_id, str) and isinstance(txt, str):
                m[_id] = txt
    return m


# -----------------------------
# Query file
# -----------------------------
@dataclass(frozen=True)
class QueryItem:
    id: str
    text: str


def load_queries_json(path: Path) -> List[QueryItem]:
    """
    Accepts either:
      - {"queries": [{"id": "...", "text": "..."}, ...]}
      - [{"id": "...", "text": "..."}, ...]
      - {"queries": ["string", "string", ...]}  (ids auto-generated)
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    items: List[QueryItem] = []

    def add_item(qid: str, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        items.append(QueryItem(id=qid, text=t))

    if isinstance(obj, dict) and "queries" in obj:
        q = obj["queries"]
        if isinstance(q, list) and q and all(isinstance(x, str) for x in q):
            for i, s in enumerate(q, start=1):
                add_item(f"q{i}", s)
            return items

        if isinstance(q, list) and q and all(isinstance(x, dict) for x in q):
            for i, d in enumerate(q, start=1):
                qid = str(d.get("id") or f"q{i}")
                text = str(d.get("text") or "")
                add_item(qid, text)
            return items

        if isinstance(q, list) and not q:
            return items

    if isinstance(obj, list):
        for i, d in enumerate(obj, start=1):
            if isinstance(d, str):
                add_item(f"q{i}", d)
            elif isinstance(d, dict):
                qid = str(d.get("id") or f"q{i}")
                text = str(d.get("text") or "")
                add_item(qid, text)
        return items

    raise ValueError(f"Unsupported queries.json structure in {path}")


# -----------------------------
# Embedding model (E5 query)
# -----------------------------
def load_hf_model(model_name: str, device: torch.device):
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: transformers. Install with: pip install transformers") from e

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModel.from_pretrained(model_name).eval().to(device)
    return mdl, tok


def embed_e5_query(
        query: str,
        *,
        model,
        tokenizer,
        device: torch.device,
        max_length: int,
) -> np.ndarray:
    """
    E5 convention for searching: prefix with 'query: '.
    Returns a float32 L2-normalized vector [D].
    """
    text = "query: " + query.strip()
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [1, T, D]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = l2_normalize(pooled).to(torch.float32).cpu().numpy()
        return pooled[0].astype(np.float32)


# -----------------------------
# Retrieval
# -----------------------------
@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    id: str
    meta: Dict[str, Any]
    text_preview: Optional[str]


def topk_cosine(emb: np.ndarray, qvec: np.ndarray, k: int) -> np.ndarray:
    """
    emb: [N, D] normalized
    qvec: [D] normalized
    returns indices of top-k scores
    """
    scores = emb @ qvec  # cosine via dot
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    # partial top-k then sort
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def make_preview(text: str, limit: int) -> str:
    t = " ".join(text.split())
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + " …"


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search top-k chunks using E5 query embeddings against embeddings.npz.")

    ap.add_argument("--embeddings", type=str, default=env_str("EMBED_OUT_NPZ", "embeddings.npz"))
    ap.add_argument("--index", type=str, default=env_str("EMBED_OUT_INDEX", "index.jsonl"))
    ap.add_argument("--prepared", type=str, default=env_str("OUT_JSONL", "prepared.jsonl"))

    ap.add_argument("--model", type=str, default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large-instruct"))
    ap.add_argument("--device", type=str, default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--query_max_length", type=int, default=env_int("QUERY_MAX_LENGTH", 256))

    ap.add_argument("--top_k", type=int, default=env_int("TOP_K", 8))
    ap.add_argument("--preview_chars", type=int, default=env_int("PREVIEW_CHARS", 450))

    ap.add_argument(
        "--queries_json",
        type=str,
        default=env_str("QUERIES_JSON", ""),
        help="Path to JSON file with example queries. If set, runs batch mode.",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (type queries). If both queries_json and interactive are set, both run.",
    )
    ap.add_argument(
        "--no_text",
        action="store_true",
        help="Do not load prepared.jsonl; only output metadata.",
    )

    return ap.parse_args()


def print_hits(qid: str, query: str, hits: List[Hit]) -> None:
    print("\n" + "=" * 90)
    print(f"QUERY {qid}: {query}")
    for h in hits:
        m = h.meta
        src = f"{m.get('source_name')} ({m.get('source_path')})"
        chunk = f"chunk_index={m.get('chunk_index')} chunk_len={m.get('chunk_len')}"
        ocr = f"pdf_ocr_used={m.get('pdf_ocr_used')} ocr_lang_used={m.get('ocr_lang_used')}"
        print(f"\n{h.rank:>2}. score={h.score:.4f}  id={h.id}")
        print(f"    {src}")
        print(f"    {chunk}  {ocr}")
        if h.text_preview:
            print(f"    text: {h.text_preview}")


def run_queries(
        queries: List[QueryItem],
        *,
        model,
        tokenizer,
        device: torch.device,
        emb: np.ndarray,
        ids: np.ndarray,
        index_rows: List[Dict[str, Any]],
        text_map: Optional[Dict[str, str]],
        top_k: int,
        preview_chars: int,
        query_max_length: int,
) -> None:
    if len(index_rows) != len(ids):
        raise RuntimeError("index.jsonl and embeddings.npz are not aligned (row count differs).")

    for qi in queries:
        qvec = embed_e5_query(
            qi.text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=query_max_length,
        )

        idxs = topk_cosine(emb, qvec, k=top_k)
        scores = (emb @ qvec)[idxs]

        hits: List[Hit] = []
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            _id = str(ids[i])
            meta = index_rows[i]
            preview = None
            if text_map is not None:
                t = text_map.get(_id, "")
                if t:
                    preview = make_preview(t, preview_chars)
            hits.append(Hit(rank=rank, score=float(s), id=_id, meta=meta, text_preview=preview))

        print_hits(qi.id, qi.text, hits)


def main() -> None:
    args = parse_args()

    emb_path = Path(args.embeddings).resolve()
    idx_path = Path(args.index).resolve()
    prep_path = Path(args.prepared).resolve()

    if not emb_path.exists():
        raise SystemExit(f"Missing embeddings file: {emb_path}")
    if not idx_path.exists():
        raise SystemExit(f"Missing index file: {idx_path}")

    ids, emb = load_npz(emb_path)
    index_rows = load_index(idx_path)

    text_map: Optional[Dict[str, str]] = None
    if not args.no_text:
        if not prep_path.exists():
            raise SystemExit(f"Missing prepared.jsonl (needed for text preview): {prep_path}")
        text_map = load_prepared_text_map(prep_path)

    device = choose_device(args.device)
    model, tokenizer = load_hf_model(args.model, device)

    # Batch mode from queries.json
    if args.queries_json:
        qpath = Path(args.queries_json).resolve()
        if not qpath.exists():
            raise SystemExit(f"queries_json not found: {qpath}")
        queries = load_queries_json(qpath)
        run_queries(
            queries,
            model=model,
            tokenizer=tokenizer,
            device=device,
            emb=emb,
            ids=ids,
            index_rows=index_rows,
            text_map=text_map,
            top_k=args.top_k,
            preview_chars=args.preview_chars,
            query_max_length=args.query_max_length,
        )

    # Interactive mode
    if args.interactive:
        print("\nInteractive search. Empty input or Ctrl-D to exit.")
        while True:
            try:
                q = input("\nQuery> ").strip()
            except EOFError:
                break
            if not q:
                break
            run_queries(
                [QueryItem(id="manual", text=q)],
                model=model,
                tokenizer=tokenizer,
                device=device,
                emb=emb,
                ids=ids,
                index_rows=index_rows,
                text_map=text_map,
                top_k=args.top_k,
                preview_chars=args.preview_chars,
                query_max_length=args.query_max_length,
            )

    if not args.queries_json and not args.interactive:
        print("Nothing to do. Provide --queries_json <file> and/or --interactive.")


if __name__ == "__main__":
    main()