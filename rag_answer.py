#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
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


def load_index_map(index_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Robust loader: id -> metadata row from index.jsonl.
    This avoids positional mismatches between ids/embeddings and index rows.
    """
    m: Dict[str, Dict[str, Any]] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in {index_path} at line {line_no}: {e}") from e
            _id = obj.get("id")
            if isinstance(_id, str):
                m[_id] = obj
    return m


def load_prepared_text_map(prepared_jsonl: Path) -> Dict[str, str]:
    """
    Loads id -> full text from prepared.jsonl.
    For ~10k chunks, keeping it in RAM is fine.
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
# Embedding model (E5 query)
# -----------------------------
def load_hf_model(model_name: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer  # type: ignore

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
class Retrieved:
    rank: int
    score: float
    id: str
    meta: Dict[str, Any]
    text: str


def topk_cosine(emb: np.ndarray, qvec: np.ndarray, k: int) -> np.ndarray:
    scores = emb @ qvec  # cosine via dot (because normalized)
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def build_context_blocks(
        hits: List[Retrieved],
        *,
        max_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Creates a context string with numbered blocks [1], [2], ...
    Also returns a structured sources list for logging/inspection.
    """
    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []
    used = 0

    for h in hits:
        m = h.meta or {}
        src = {
            "n": h.rank,
            "score": round(h.score, 4),
            "id": h.id,
            "source_name": m.get("source_name") or m.get("origin_source_name"),
            "source_path": m.get("source_path") or m.get("origin_source_path"),
            "chunk_index": m.get("chunk_index"),
            "chunk_len": m.get("chunk_len"),
            "slide_index": m.get("slide_index"),
            "pdf_ocr_used": m.get("pdf_ocr_used"),
            "ocr_lang_used": m.get("ocr_lang_used"),
            "vision_model": m.get("vision_model"),
        }
        sources.append(src)

        file_name = m.get("source_name") or m.get("origin_source_name") or "?"
        chunk_idx = m.get("chunk_index")
        slide_idx = m.get("slide_index")

        extra = []
        if slide_idx is not None:
            extra.append(f"slide_index={slide_idx}")
        if chunk_idx is not None:
            extra.append(f"chunk_index={chunk_idx}")
        extra_s = (" " + " ".join(extra)) if extra else ""

        header = f"[{h.rank}] score={h.score:.4f} file={file_name}{extra_s} id={h.id}"

        body = (h.text or "").strip()
        if not body:
            continue

        block = header + "\n" + body
        if used + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - used - len(header) - 2)
            if remaining > 0:
                block = header + "\n" + body[:remaining].rstrip() + "\n…"
                blocks.append(block)
            break

        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks).strip(), sources


# -----------------------------
# Generic LLM client (clean env-driven)
# -----------------------------
class LLMClient:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


def require_env(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return v


def env_int_optional(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    return int(v) if v else default


def env_json_object_optional(key: str) -> Dict[str, Any]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{key} must be a JSON object, got {type(obj).__name__}")
    return obj


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, options: Dict[str, Any], timeout_s: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = options
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if self.options:
            payload["options"] = self.options

        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return content.strip()


def make_llm_client() -> LLMClient:
    backend = require_env("LLM_BACKEND").lower()
    model = require_env("LLM_MODEL")
    timeout_s = env_int_optional("LLM_TIMEOUT_S", 300)
    options = env_json_object_optional("LLM_OPTIONS_JSON")

    if backend == "ollama":
        base_url = require_env("OLLAMA_BASE_URL")
        return OllamaClient(base_url=base_url, model=model, options=options, timeout_s=timeout_s)

    raise RuntimeError(f"Unsupported LLM_BACKEND: {backend}")


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "Du bist ein präziser Assistent. Beantworte die FRAGE ausschliesslich anhand des KONTEXT.\n"
        "Wenn der KONTEXT nicht ausreicht, sage das explizit und nenne, welche Information fehlt.\n"
        "Zitiere Aussagen mit [1], [2], ... entsprechend den Kontext-Blöcken."
    )
    user = f"FRAGE:\n{question.strip()}\n\nKONTEXT:\n{context}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -----------------------------
# CLI / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAG: retrieve relevant chunks and answer with an env-configured LLM.")

    ap.add_argument("--question", type=str, default="", help="If empty: interactive mode.")
    ap.add_argument("--top_k", type=int, default=env_int("TOP_K", 8))
    ap.add_argument("--context_max_chars", type=int, default=env_int("CONTEXT_MAX_CHARS", 12000))
    ap.add_argument("--print_sources", action="store_true", help="Print sources metadata as JSON.")
    ap.add_argument("--print_context", action="store_true", help="Print retrieved context blocks.")

    ap.add_argument("--embeddings", type=str, default=env_str("EMBED_OUT_NPZ", "embeddings.npz"))
    ap.add_argument("--index", type=str, default=env_str("EMBED_OUT_INDEX", "index.jsonl"))
    ap.add_argument("--prepared", type=str, default=env_str("OUT_JSONL", "prepared.jsonl"))

    ap.add_argument("--embed_model", type=str, default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large-instruct"))
    ap.add_argument("--embed_device", type=str, default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--query_max_length", type=int, default=env_int("QUERY_MAX_LENGTH", 256))

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    emb_path = Path(args.embeddings).resolve()
    idx_path = Path(args.index).resolve()
    prep_path = Path(args.prepared).resolve()

    if not emb_path.exists():
        raise SystemExit(f"Missing embeddings file: {emb_path}")
    if not idx_path.exists():
        raise SystemExit(f"Missing index file: {idx_path}")
    if not prep_path.exists():
        raise SystemExit(f"Missing prepared.jsonl: {prep_path}")

    ids, emb = load_npz(emb_path)
    index_map = load_index_map(idx_path)
    text_map = load_prepared_text_map(prep_path)

    device = choose_device(args.embed_device)
    embed_model, embed_tok = load_hf_model(args.embed_model, device)

    llm = make_llm_client()

    def answer(question: str) -> None:
        qvec = embed_e5_query(
            question,
            model=embed_model,
            tokenizer=embed_tok,
            device=device,
            max_length=args.query_max_length,
        )

        idxs = topk_cosine(emb, qvec, k=args.top_k)
        scores = (emb @ qvec)[idxs]

        hits: List[Retrieved] = []
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            _id = str(ids[i])
            meta = index_map.get(_id, {})
            text = text_map.get(_id, "")
            hits.append(Retrieved(rank=rank, score=float(s), id=_id, meta=meta, text=text))

        context, sources = build_context_blocks(hits, max_chars=args.context_max_chars)

        if args.print_sources:
            print(json.dumps({"question": question, "sources": sources}, ensure_ascii=False, indent=2))

        if args.print_context:
            print("\n" + "=" * 90)
            print("RETRIEVED CONTEXT")
            print("=" * 90)
            print(context)

        messages = build_messages(question, context)
        reply = llm.chat(messages)

        print("\n" + "=" * 90)
        print("ANSWER")
        print("=" * 90)
        print(reply)

    if args.question.strip():
        answer(args.question)
        return

    print("Interactive RAG. Empty input to exit.")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            break
        answer(q)


if __name__ == "__main__":
    main()