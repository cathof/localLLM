#!/usr/bin/env python3
from __future__ import annotations

# ── Usage ─────────────────────────────────────────────────────────────────────
#
# Simple Q&A (original mode, unchanged):
#   python rag_answer.py --question "Wie minimiere ich die Kontamination?" \
#                        --print_sources --print_context
#
# Interactive Q&A (original mode, unchanged):
#   python rag_answer.py
#
# Document error detection (new mode):
#   python rag_answer.py --document path/to/MyDoc.docx \
#                        --print_sources --print_context
#
# Dual RAG (writing rules + additional material):
#   python rag_answer.py --document path/to/MyDoc.docx \
#       --embeddings      embeddings_rules.npz \
#       --index           index_rules.jsonl \
#       --prepared        prepared_rules.jsonl \
#       --embeddings2     embeddings_material.npz \
#       --index2          index_material.jsonl \
#       --prepared2       prepared_material.jsonl
#
# All settings can also be provided via .env (see defaults below).
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch


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


load_dotenv(".env")


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() != "" else default

def env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    return int(v) if v else default

def require_env(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return v

def env_json_object_optional(key: str) -> Dict[str, Any]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{key} must be a JSON object, got {type(obj).__name__}")
    return obj


# ── Device + math ─────────────────────────────────────────────────────────────

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


# ── RAG store ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagStore:
    """
    One complete RAG index: embeddings, metadata index, and text lookup.

    name        short label used in log output ("rules" or "material")
    ids         numpy array of chunk IDs, aligned with emb rows
    emb         float32 numpy array [N, D] of L2-normalised embeddings
    index_map   id → metadata dict loaded from index.jsonl
    text_map    id → chunk text loaded from prepared.jsonl
    """
    name:      str
    ids:       np.ndarray
    emb:       np.ndarray
    index_map: Dict[str, Dict[str, Any]]
    text_map:  Dict[str, str]


# ── Artifact loaders ──────────────────────────────────────────────────────────

def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    ids  = data["ids"]
    emb  = data["embeddings"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    if len(ids) != emb.shape[0]:
        raise ValueError(f"ids length {len(ids)} != embeddings rows {emb.shape[0]}")
    return ids, emb


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {e}") from e


def load_index_map(index_path: Path) -> Dict[str, Dict[str, Any]]:
    """id → metadata row from index.jsonl — keyed by id for robust lookup."""
    m: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(index_path):
        _id = obj.get("id")
        if isinstance(_id, str):
            m[_id] = obj
    return m


def load_prepared_text_map(prepared_jsonl: Path) -> Dict[str, str]:
    """id → chunk text from prepared.jsonl."""
    m: Dict[str, str] = {}
    for obj in _iter_jsonl(prepared_jsonl):
        _id = obj.get("id")
        txt = obj.get("text")
        if isinstance(_id, str) and isinstance(txt, str):
            m[_id] = txt
    return m


def load_rag_store(
        name: str,
        npz_path: Path,
        index_path: Path,
        prepared_path: Path,
) -> RagStore:
    """Load and validate one complete RAG store from disk."""
    if not npz_path.exists():
        raise SystemExit(f"[{name}] Embeddings not found: {npz_path}")
    if not index_path.exists():
        raise SystemExit(f"[{name}] Index not found: {index_path}")
    if not prepared_path.exists():
        raise SystemExit(f"[{name}] Prepared JSONL not found: {prepared_path}")

    ids, emb  = load_npz(npz_path)
    index_map = load_index_map(index_path)
    text_map  = load_prepared_text_map(prepared_path)

    print(
        f"[INFO] Loaded RAG store '{name}': "
        f"{emb.shape[0]} chunks, dim={emb.shape[1]}, "
        f"index={len(index_map)}, texts={len(text_map)}"
    )
    return RagStore(
        name=name, ids=ids, emb=emb,
        index_map=index_map, text_map=text_map,
    )


# ── Embedding model (E5 query) ────────────────────────────────────────────────

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
    Embed a retrieval query using the E5 convention.
    The 'query: ' prefix is mandatory — omitting it significantly degrades
    retrieval quality against passages indexed with 'passage: '.
    Returns a float32 L2-normalised vector [D].
    """
    text = "query: " + query.strip()
    enc  = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out         = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask        = attention_mask.unsqueeze(-1).type_as(last_hidden)
        pooled      = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled      = l2_normalize(pooled).to(torch.float32).cpu().numpy()
    return pooled[0].astype(np.float32)


# ── Retrieval ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Retrieved:
    rank:  int
    score: float
    id:    str
    meta:  Dict[str, Any]
    text:  str
    vec:   Optional[np.ndarray] = None
    retrieval_query: str = ""


def topk_cosine(emb: np.ndarray, qvec: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k rows by cosine similarity (dot product, normalised)."""
    scores = emb @ qvec
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


_STOPWORDS_DE = {
    "als", "am", "an", "auch", "aus", "bei", "bis", "dabei", "das", "dass", "dem", "den",
    "der", "des", "die", "dies", "diese", "dieser", "doch", "durch", "ein", "eine", "einem",
    "einen", "einer", "eines", "er", "es", "für", "habe", "haben", "hat", "hinter", "ich",
    "im", "in", "ist", "ja", "kann", "können", "mich", "mir", "mit", "muss", "müssen", "nach",
    "noch", "nun", "oder", "sehr", "sein", "sind", "so", "soll", "sollen", "tue", "tun", "und",
    "unter", "vom", "von", "vor", "war", "was", "welche", "welcher", "welches", "wenn", "wer",
    "wie", "wir", "wird", "wo", "zu", "zum", "zur",
}


def _normalize_query_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()



def _extract_query_keywords(text: str, *, max_tokens: int = 8) -> str:
    raw_tokens = [t.lower() for t in __import__('re').findall(r"[A-Za-zÄÖÜäöüß0-9][A-Za-zÄÖÜäöüß0-9\-_/]+", text)]
    keep: List[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        if len(tok) < 4 or tok in _STOPWORDS_DE or tok in seen:
            continue
        seen.add(tok)
        keep.append(tok)
        if len(keep) >= max_tokens:
            break
    return " ".join(keep)


def build_multi_queries(query_text: str, *, mode: str, max_queries: int) -> List[str]:
    """
    Create a small set of internal retrieval queries from one user query.

    The goal is recall and aspect coverage, not linguistic perfection.
    This stays intentionally simple and deterministic:
      - original query
      - keyword-compressed variant
      - generic task-oriented reformulations for process questions
    """
    base = _normalize_query_text(query_text)
    if not base:
        return []

    variants: List[str] = [base]
    keywords = _extract_query_keywords(base)
    if keywords and keywords != base.lower():
        variants.append(keywords)

    base_l = base.lower()
    process_cues = ("wie ", "vorgehen", "ablauf", "prozess", "schritte", "checkliste")
    if keywords and (mode == "segment" or any(c in base_l for c in process_cues)):
        variants.extend([
            f"{keywords} vorgehen",
            f"{keywords} schritte",
            f"{keywords} checkliste",
        ])

    deduped: List[str] = []
    seen: set[str] = set()
    for q in variants:
        qn = _normalize_query_text(q)
        key = qn.lower()
        if not qn or key in seen:
            continue
        seen.add(key)
        deduped.append(qn)
        if len(deduped) >= max_queries:
            break
    return deduped


def retrieve_from_store(
        store: RagStore,
        qvec: np.ndarray,
        k: int,
        *,
        retrieval_query: str = "",
) -> List[Retrieved]:
    """Query a single RAG store and return top-k hits enriched with vectors for MMR."""
    idxs   = topk_cosine(store.emb, qvec, k=k)
    scores = (store.emb @ qvec)[idxs]
    hits: List[Retrieved] = []
    for i, s in zip(idxs, scores):
        _id  = str(store.ids[i])
        meta = store.index_map.get(_id, {})
        text = store.text_map.get(_id, "")
        hits.append(Retrieved(
            rank=0,
            score=float(s),
            id=_id,
            meta=meta,
            text=text,
            vec=store.emb[i],
            retrieval_query=retrieval_query,
        ))
    return hits


def merge_hits(
        hits_per_store: List[List[Retrieved]],
        *,
        top_k: int,
) -> List[Retrieved]:
    """
    Merge retrieved hits from multiple RAG stores.

    Deduplicates by chunk id, sorts by score descending, truncates to
    top_k, then assigns final sequential ranks starting from 1.
    """
    seen:   set[str]        = set()
    merged: List[Retrieved] = []
    for hits in hits_per_store:
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                merged.append(h)

    merged.sort(key=lambda h: h.score, reverse=True)
    merged = merged[:top_k]

    return [
        Retrieved(
            rank=i + 1,
            score=h.score,
            id=h.id,
            meta=h.meta,
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for i, h in enumerate(merged)
    ]


def _source_key(hit: Retrieved) -> str:
    meta = hit.meta or {}
    return str(
        meta.get("source_path")
        or meta.get("origin_source_path")
        or meta.get("source_name")
        or meta.get("origin_source_name")
        or ""
    )


def _max_similarity_to_selected(hit: Retrieved, selected: Sequence[Retrieved]) -> float:
    if hit.vec is None or not selected:
        return 0.0
    sims: List[float] = []
    for other in selected:
        if other.vec is None:
            continue
        sims.append(float(hit.vec @ other.vec))
    return max(sims) if sims else 0.0


def diversify_hits_mmr(
        candidates: List[Retrieved],
        *,
        top_k: int,
        mmr_lambda: float,
        max_per_source: int,
) -> List[Retrieved]:
    """
    Select a diverse final context with a simple MMR strategy.

    - keeps high-relevance chunks
    - penalises near-duplicates
    - optionally caps how many chunks come from the same source file
    """
    unique_by_id: Dict[str, Retrieved] = {}
    for cand in candidates:
        prev = unique_by_id.get(cand.id)
        if prev is None or cand.score > prev.score:
            unique_by_id[cand.id] = cand

    pool = sorted(unique_by_id.values(), key=lambda h: h.score, reverse=True)
    if not pool:
        return []

    selected: List[Retrieved] = []
    source_counts: Dict[str, int] = {}

    def choose_candidate(enforce_source_cap: bool) -> Optional[Retrieved]:
        best_hit: Optional[Retrieved] = None
        best_mmr = float("-inf")
        for cand in pool:
            if cand.id in {s.id for s in selected}:
                continue
            src_key = _source_key(cand)
            if enforce_source_cap and max_per_source > 0 and source_counts.get(src_key, 0) >= max_per_source:
                continue
            redundancy = _max_similarity_to_selected(cand, selected)
            mmr_score = (mmr_lambda * cand.score) - ((1.0 - mmr_lambda) * redundancy)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_hit = cand
        return best_hit

    while len(selected) < min(top_k, len(pool)):
        chosen = choose_candidate(enforce_source_cap=True)
        if chosen is None:
            chosen = choose_candidate(enforce_source_cap=False)
        if chosen is None:
            break
        selected.append(chosen)
        source_counts[_source_key(chosen)] = source_counts.get(_source_key(chosen), 0) + 1

    return [
        Retrieved(
            rank=i + 1,
            score=h.score,
            id=h.id,
            meta=h.meta,
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for i, h in enumerate(selected)
    ]


def retrieve_multi_query(
        query_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        max_length: int,
        top_k: int,
        candidate_k: int,
        multi_query_count: int,
        mmr_lambda: float,
        max_per_source: int,
        mode: str,
) -> Tuple[List[Retrieved], List[str]]:
    """
    Run internal multi-query retrieval and select a diverse final context.
    """
    queries = build_multi_queries(query_text, mode=mode, max_queries=multi_query_count)
    all_candidates: List[Retrieved] = []

    for query_variant in queries:
        qvec = embed_e5_query(
            query_variant,
            model=embed_model,
            tokenizer=embed_tok,
            device=device,
            max_length=max_length,
        )
        for store in stores:
            all_candidates.extend(
                retrieve_from_store(
                    store,
                    qvec,
                    k=candidate_k,
                    retrieval_query=query_variant,
                )
            )

    selected = diversify_hits_mmr(
        all_candidates,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        max_per_source=max_per_source,
    )
    return selected, queries


# ── Lazy image captioning ─────────────────────────────────────────────────────

def enrich_hits_with_image_captions(
        hits: List[Retrieved],
        *,
        vision_cfg: dict,
        max_workers: int = 3,
) -> List[Retrieved]:
    """
    For each retrieved chunk that has embedded images, call qwen2.5vl:7b
    via Ollama to generate a caption and append it to the chunk text.

    Calls run in parallel (up to max_workers) to reduce wall time.
    Chunks without images pass through unchanged.

    vision_cfg must contain at minimum:
        vision_model      e.g. "qwen2.5vl:7b"
        ollama_base_url   e.g. "http://localhost:11434"
        vision_timeout_s  e.g. 180

    Import:
        from importDocuments import ollama_caption_png
    """
    try:
        from importDocuments import ollama_caption_png
    except ImportError:
        print("[WARN] importDocuments not found — skipping image captioning")
        return hits

    def process(hit: Retrieved) -> Retrieved:
        image_paths: List[str] = hit.meta.get("embedded_images") or []
        if not image_paths:
            return hit

        captions: List[str] = []
        for img_path in image_paths:
            p = Path(img_path)
            if not p.exists():
                continue
            try:
                caption = ollama_caption_png(cfg=vision_cfg, png_bytes=p.read_bytes())
                if caption:
                    captions.append(caption)
            except Exception as e:
                print(f"[WARN] Vision captioning failed for {p.name}: {e}")
                continue

        if not captions:
            return hit

        enriched_text = hit.text.rstrip()
        for i, cap in enumerate(captions, start=1):
            enriched_text += f"\n\n[Bild {i}]: {cap}"

        return Retrieved(
            rank=hit.rank, score=hit.score,
            id=hit.id, meta=hit.meta,
            text=enriched_text,
            vec=hit.vec,
            retrieval_query=hit.retrieval_query,
        )

    hits_with    = [h for h in hits if h.meta.get("embedded_images")]
    hits_without = [h for h in hits if not h.meta.get("embedded_images")]

    enriched = list(hits_without)

    if hits_with:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process, h): h for h in hits_with}
            for future in as_completed(futures):
                enriched.append(future.result())
        enriched.sort(key=lambda h: h.rank)

    return enriched


# ── Context builder ───────────────────────────────────────────────────────────

def build_context_blocks(
        hits: List[Retrieved],
        *,
        max_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a numbered context string [1], [2], ... from retrieved hits.
    Also returns a structured sources list for logging / inspection.
    Respects max_chars: truncates the last block rather than dropping it.
    """
    sources: List[Dict[str, Any]] = []
    blocks:  List[str]            = []
    used = 0

    for h in hits:
        m = h.meta or {}

        src = {
            "n":               h.rank,
            "score":           round(h.score, 4),
            "id":              h.id,
            "retrieval_query": h.retrieval_query,
            "source_name":     m.get("source_name") or m.get("origin_source_name"),
            "source_path":     m.get("source_path") or m.get("origin_source_path"),
            "chunk_index":     m.get("chunk_index"),
            "chunk_len":       m.get("chunk_len"),
            "pdf_ocr_used":    m.get("pdf_ocr_used"),
            "pdf_text_reader": m.get("pdf_text_reader"),    # which library extracted text
            "ocr_lang":        m.get("ocr_lang"),           # renamed from ocr_lang_used
            "embedded_images": m.get("embedded_images", []), # image cache paths
        }
        sources.append(src)

        file_name = m.get("source_name") or m.get("origin_source_name") or "?"
        extra = []
        if m.get("chunk_index") is not None:
            extra.append(f"chunk_index={m['chunk_index']}")
        if m.get("section_title"):
            extra.append(f"section={m['section_title']!r}")
        extra_s = (" " + " ".join(extra)) if extra else ""

        header = f"[{h.rank}] score={h.score:.4f} file={file_name}{extra_s} id={h.id}"
        body   = (h.text or "").strip()
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


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(
            self,
            base_url: str,
            model: str,
            options: Dict[str, Any],
            timeout_s: int,
    ):
        self.base_url  = base_url.rstrip("/")
        self.model     = model
        self.options   = options
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url     = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
        }
        if self.options:
            payload["options"] = self.options

        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data    = r.json()
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return content.strip()


def make_llm_client() -> LLMClient:
    backend   = require_env("LLM_BACKEND").lower()
    model     = require_env("LLM_MODEL")
    timeout_s = env_int("LLM_TIMEOUT_S", 300)
    options   = env_json_object_optional("LLM_OPTIONS_JSON")

    if backend == "ollama":
        base_url = require_env("OLLAMA_BASE_URL")
        return OllamaClient(
            base_url=base_url, model=model,
            options=options, timeout_s=timeout_s,
        )

    raise RuntimeError(f"Unsupported LLM_BACKEND: {backend!r}")


# ── Prompts ───────────────────────────────────────────────────────────────────

def build_qa_messages(
        question: str,
        context: str,
) -> List[Dict[str, str]]:
    """
    General-purpose RAG Q&A prompt.
    Used by the --question and interactive modes (original behaviour).
    """
    system = (
        "Du bist ein präziser Assistent für transparente RAG-Antworten.\n"
        "Beantworte die FRAGE ausschliesslich anhand des KONTEXTS.\n"
        "Nutze alle relevanten Informationen aus dem KONTEXT möglichst vollständig.\n"
        "Erfinde nichts und ergänze nichts aus eigenem Wissen.\n"
        "Strukturiere die Antwort in thematische Schritte oder Phasen.\n"
        "Jede inhaltliche Aussage muss mit [1], [2], ... belegt werden.\n"
        "Wenn der KONTEXT nur Teilaspekte enthält, sage klar, welche Teile fehlen."
    )
    user = (
        f"FRAGE:\n{question.strip()}\n\n"
        f"KONTEXT:\n{context}\n\n"
        "Erstelle eine ausführliche, quellennahe Antwort. "
        "Übernimm möglichst viele konkrete Punkte aus dem Kontext, "
        "ohne sie unnötig zu paraphrasieren."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_error_detection_messages(
        document_text: str,
        context: str,
) -> List[Dict[str, str]]:
    """
    Error detection prompt for qwen2.5:14b-instruct.
    Used by the --document mode.

    The context contains retrieved chunks from:
      - RAG 1 (writing rules / Regelwerk)
      - RAG 2 (additional material / Quellmaterial)

    The model must detect and classify errors in the document
    using only what appears in the context.
    """
    system = (
        "Du bist ein präziser Qualitätsprüfer für technische Dokumente.\n"
        "Dir wird ein zu prüfendes DOKUMENT und ein KONTEXT aus Referenzmaterial "
        "und Regelwerk vorgelegt.\n"
        "Deine Aufgabe: Erkenne alle Fehler im DOKUMENT anhand des KONTEXTS.\n\n"
        "Fehlerklassen:\n"
        "  Inhaltlicher Fehler    – Aussage widerspricht dem Referenzmaterial\n"
        "  Fehlende Information   – Pflichtangabe fehlt laut Regelwerk\n"
        "  Formaler Fehler        – Formatierung, Struktur oder Benennung entspricht nicht den Vorgaben\n"
        "  Veraltete Information  – Wert oder Referenz stimmt nicht mit aktuellem Quellmaterial überein\n\n"
        "Regeln:\n"
        "  - Belege jeden Fehler mit [N] aus dem KONTEXT\n"
        "  - Wenn kein Fehler erkennbar ist: 'Kein Fehler gefunden'\n"
        "  - Erfinde nichts, das nicht im KONTEXT steht\n"
        "  - Ausgabe als nummerierte Fehlerliste:\n"
        "    1. [Fehlerklasse] Betroffene Stelle | Begründung | Quelle [N]"
    )
    user = (
        f"DOKUMENT ZUR PRÜFUNG:\n{document_text.strip()}\n\n"
        f"KONTEXT (Regelwerk + Referenzmaterial):\n{context}\n\n"
        "Erstelle eine vollständige Fehlerliste. "
        "Für jeden Fehler: Fehlerklasse, betroffene Stelle im Dokument, "
        "Begründung mit Quelle [N]."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ── Document segmentation for segment-level retrieval ─────────────────────────

def split_document_into_segments(
        document_text: str,
        *,
        target_chars: int = 1200,
        min_chars: int = 250,
        max_chars: int = 2200,
) -> List[str]:
    """
    Split a full document into retrieval-friendly segments.

    Strategy:
      1. Preserve double-newline paragraph boundaries
      2. Build medium-sized segments so each query stays focused
      3. Merge tiny trailing fragments into neighbours

    This is used only for document checking. The final report still evaluates
    the entire document, but retrieval happens per segment instead of once for
    the whole document.
    """
    text = (document_text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]

    segments: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        segment = "\n\n".join(current).strip()
        if segment:
            segments.append(segment)
        current = []
        current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len >= max_chars:
            if current:
                flush()
            start = 0
            while start < para_len:
                end = min(para_len, start + target_chars)
                chunk = para[start:end].strip()
                if chunk:
                    segments.append(chunk)
                start = end
            continue

        candidate_len = current_len + (2 if current else 0) + para_len
        if current and candidate_len > target_chars:
            flush()

        current.append(para)
        current_len = len("\n\n".join(current))

        if current_len >= target_chars:
            flush()

    flush()

    merged: List[str] = []
    for seg in segments:
        if merged and len(seg) < min_chars:
            candidate = merged[-1] + "\n\n" + seg
            if len(candidate) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(seg)

    return merged or [text]


def retrieve_document_context_by_segments(
        document_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
        per_segment_k: int = 3,
) -> Tuple[str, List[Retrieved], List[str], List[str]]:
    """
    Segment-level retrieval for document checking.

    Each segment is queried separately. Within each segment, internal
    multi-query retrieval broadens recall; afterwards MMR keeps the final
    document context compact and non-redundant.
    """
    segments = split_document_into_segments(document_text)
    all_candidates: List[Retrieved] = []
    all_queries: List[str] = []

    per_segment_candidate_k = max(per_segment_k, min(max(args.top_k, 8), per_segment_k * 4))

    for segment in segments:
        hits, queries = retrieve_multi_query(
            segment,
            stores,
            embed_model=embed_model,
            embed_tok=embed_tok,
            device=device,
            max_length=args.query_max_length,
            top_k=per_segment_k,
            candidate_k=per_segment_candidate_k,
            multi_query_count=max(2, min(args.multi_query_count, 3)),
            mmr_lambda=args.mmr_lambda,
            max_per_source=max(1, args.max_per_source),
            mode="segment",
        )
        all_candidates.extend(hits)
        all_queries.extend(queries)

    ranked = diversify_hits_mmr(
        all_candidates,
        top_k=args.top_k,
        mmr_lambda=args.mmr_lambda,
        max_per_source=args.max_per_source,
    )

    if vision_cfg and vision_cfg.get("vision_model"):
        ranked = enrich_hits_with_image_captions(
            ranked,
            vision_cfg=vision_cfg,
            max_workers=args.vision_workers,
        )

    context, _sources = build_context_blocks(ranked, max_chars=args.context_max_chars)
    deduped_queries = list(dict.fromkeys(q for q in all_queries if q))
    return context, ranked, segments, deduped_queries


# ── Shared retrieval + output helper ─────────────────────────────────────────

def _retrieve_and_print(
        query_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
        label: str = "ANSWER",
        log_key: str = "question",
) -> Tuple[str, List[Retrieved], List[str]]:
    """
    Shared retrieval pipeline used by both Q&A and document checking.

    Retrieval strategy:
      1. create a few internal query variants
      2. retrieve a broader candidate pool
      3. select a diverse final context with MMR
      4. optionally enrich with vision captions
    """
    candidate_k = max(args.top_k, min(max(args.top_k * 4, 12), 40))
    hits, queries = retrieve_multi_query(
        query_text,
        stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        max_length=args.query_max_length,
        top_k=args.top_k,
        candidate_k=candidate_k,
        multi_query_count=args.multi_query_count,
        mmr_lambda=args.mmr_lambda,
        max_per_source=args.max_per_source,
        mode="qa",
    )

    if vision_cfg and vision_cfg.get("vision_model"):
        hits = enrich_hits_with_image_captions(
            hits,
            vision_cfg=vision_cfg,
            max_workers=args.vision_workers,
        )

    context, sources = build_context_blocks(hits, max_chars=args.context_max_chars)

    if args.print_sources:
        print(json.dumps({
            log_key: query_text[:120],
            "multi_queries": queries,
            "sources": sources,
        }, ensure_ascii=False, indent=2))

    if args.print_context:
        print("\n" + "=" * 90)
        print("RETRIEVED CONTEXT")
        print("=" * 90)
        print(context)

    return context, hits, queries


# ── Mode: Q&A (original) ─────────────────────────────────────────────────────

def answer(
        question: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        llm: LLMClient,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
) -> None:
    """
    Original Q&A mode — unchanged behaviour.
    Retrieves relevant chunks and answers the question with the LLM.
    """
    context, _, _ = _retrieve_and_print(
        question, stores,
        embed_model=embed_model, embed_tok=embed_tok,
        device=device, args=args, vision_cfg=vision_cfg,
        label="ANSWER", log_key="question",
    )

    messages = build_qa_messages(question, context)
    reply    = llm.chat(messages)

    print("\n" + "=" * 90)
    print("ANSWER")
    print("=" * 90)
    print(reply)


# ── Mode: document error detection (new) ─────────────────────────────────────

def check_document(
        doc_path: Path,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        llm: LLMClient,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
) -> None:
    """
    Document error detection mode using segment-level retrieval.

    The document is still evaluated as a whole, but retrieval is done per
    segment so later sections and local issues are less likely to be missed.
    The retrieved evidence is then merged into one final context for the
    document-level checking prompt.
    """
    try:
        from importDocuments import normalize_text, read_docx
    except ImportError as e:
        raise SystemExit(
            "importDocuments.py must be in the same directory or on PYTHONPATH. "
            f"Original error: {e}"
        )

    print(f"[INFO] Reading document: {doc_path.name}")
    doc_text = normalize_text(read_docx(doc_path))

    if not doc_text.strip():
        print(f"[WARN] No text extracted from {doc_path.name} — aborting.")
        return

    print(f"[INFO] Document text: {len(doc_text)} chars")

    per_segment_k = max(1, min(3, args.top_k))
    context, hits, segments, multi_queries = retrieve_document_context_by_segments(
        doc_text, stores,
        embed_model=embed_model, embed_tok=embed_tok,
        device=device, args=args, vision_cfg=vision_cfg,
        per_segment_k=per_segment_k,
    )

    if args.print_sources:
        sources = []
        for h in hits:
            m = h.meta or {}
            sources.append({
                "n": h.rank,
                "score": round(h.score, 4),
                "id": h.id,
                "retrieval_query": h.retrieval_query,
                "source_name": m.get("source_name") or m.get("origin_source_name"),
                "source_path": m.get("source_path") or m.get("origin_source_path"),
                "chunk_index": m.get("chunk_index"),
                "chunk_len": m.get("chunk_len"),
                "pdf_ocr_used": m.get("pdf_ocr_used"),
                "pdf_text_reader": m.get("pdf_text_reader"),
                "ocr_lang": m.get("ocr_lang"),
                "embedded_images": m.get("embedded_images", []),
            })
        print(json.dumps({
            "document": str(doc_path),
            "document_chars": len(doc_text),
            "segments": len(segments),
            "multi_queries": multi_queries,
            "sources": sources,
        }, ensure_ascii=False, indent=2))

    if args.print_context:
        print("\n" + "=" * 90)
        print("RETRIEVED CONTEXT")
        print("=" * 90)
        print(context)

    messages = build_error_detection_messages(doc_text, context)
    reply = llm.chat(messages)

    print("\n" + "=" * 90)
    print(f"ERROR DETECTION REPORT — {doc_path.name}")
    print("=" * 90)
    print(reply)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "RAG pipeline with dual-store retrieval.\n\n"
            "Modes:\n"
            "  --question TEXT   Answer a free-text question (original mode)\n"
            "  (no args)         Interactive Q&A loop (original mode)\n"
            "  --document FILE   Detect errors in a Word document (new mode)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode selection ─────────────────────────────────────────────────────────
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--question", type=str, default="",
        help="Free-text question; if empty, starts interactive loop",
    )
    mode.add_argument(
        "--document", type=str, default="",
        help="Path to a .docx file to check for errors",
    )

    # ── RAG store 1 (writing rules — required) ─────────────────────────────────
    ap.add_argument("--embeddings", type=str,
                    default=env_str("EMBED_OUT_NPZ",   "embeddings.npz"),
                    help="Embeddings .npz for RAG store 1 (writing rules)")
    ap.add_argument("--index", type=str,
                    default=env_str("EMBED_OUT_INDEX", "index.jsonl"),
                    help="Index .jsonl for RAG store 1")
    ap.add_argument("--prepared", type=str,
                    default=env_str("OUT_JSONL",       "prepared.jsonl"),
                    help="Prepared .jsonl for RAG store 1")

    # ── RAG store 2 (additional material — optional) ───────────────────────────
    ap.add_argument("--embeddings2", type=str,
                    default=env_str("EMBED_OUT_NPZ2",   ""),
                    help="Embeddings .npz for RAG store 2 (additional material)")
    ap.add_argument("--index2", type=str,
                    default=env_str("EMBED_OUT_INDEX2", ""),
                    help="Index .jsonl for RAG store 2")
    ap.add_argument("--prepared2", type=str,
                    default=env_str("OUT_JSONL2",       ""),
                    help="Prepared .jsonl for RAG store 2")

    # ── Retrieval settings ─────────────────────────────────────────────────────
    ap.add_argument("--top_k", type=int,
                    default=env_int("TOP_K", 8),
                    help="Total top-k chunks to retrieve (across all stores)")
    ap.add_argument("--context_max_chars", type=int,
                    default=env_int("CONTEXT_MAX_CHARS", 12000),
                    help="Maximum characters in the context passed to the LLM")

    # ── Embedding model ────────────────────────────────────────────────────────
    ap.add_argument("--embed_model", type=str,
                    default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large-instruct"))
    ap.add_argument("--embed_device", type=str,
                    default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--query_max_length", type=int,
                    default=env_int("QUERY_MAX_LENGTH", 256))
    ap.add_argument("--multi_query_count", type=int,
                    default=env_int("MULTI_QUERY_COUNT", 4),
                    help="Number of internal retrieval query variants to generate")
    ap.add_argument("--mmr_lambda", type=float,
                    default=float(env_str("MMR_LAMBDA", "0.75")),
                    help="MMR relevance weight between 0 and 1; lower means more diversification")
    ap.add_argument("--max_per_source", type=int,
                    default=env_int("MAX_PER_SOURCE", 2),
                    help="Maximum number of final context chunks per source file")

    # ── Vision (lazy inference-time captioning) ────────────────────────────────
    ap.add_argument("--vision_model", type=str,
                    default=env_str("VISION_MODEL", ""),
                    help="Ollama vision model for lazy image captioning, e.g. qwen2.5vl:7b. "
                         "Leave empty to skip captioning.")
    ap.add_argument("--vision_workers", type=int,
                    default=env_int("VISION_WORKERS", 3),
                    help="Parallel workers for image captioning")
    ap.add_argument("--vision_timeout_s", type=int,
                    default=env_int("VISION_TIMEOUT_S", 180))

    # ── Output flags ───────────────────────────────────────────────────────────
    ap.add_argument("--print_sources", action="store_true",
                    help="Print sources metadata as JSON")
    ap.add_argument("--print_context", action="store_true",
                    help="Print retrieved context blocks before the answer")

    return ap.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load RAG store 1 (always required) ────────────────────────────────────
    stores: List[RagStore] = [
        load_rag_store(
            "rules",
            npz_path=Path(args.embeddings).resolve(),
            index_path=Path(args.index).resolve(),
            prepared_path=Path(args.prepared).resolve(),
        )
    ]

    # ── Load RAG store 2 (optional — additional material) ─────────────────────
    if args.embeddings2.strip():
        stores.append(
            load_rag_store(
                "material",
                npz_path=Path(args.embeddings2).resolve(),
                index_path=Path(args.index2).resolve(),
                prepared_path=Path(args.prepared2).resolve(),
            )
        )
    else:
        print("[INFO] No second RAG store configured — using rules store only.")

    # ── Embedding model ────────────────────────────────────────────────────────
    device = choose_device(args.embed_device)
    print(f"[INFO] Embedding device: {device} | model: {args.embed_model}")
    embed_model, embed_tok = load_hf_model(args.embed_model, device)

    # ── LLM client ─────────────────────────────────────────────────────────────
    llm = make_llm_client()

    # ── Vision config (passed through to lazy captioning) ─────────────────────
    vision_cfg: Optional[dict] = None
    if args.vision_model.strip():
        vision_cfg = {
            "vision_model":     args.vision_model,
            "ollama_base_url":  env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            "vision_timeout_s": args.vision_timeout_s,
        }
        print(f"[INFO] Vision captioning enabled: {args.vision_model}")
    else:
        print("[INFO] Vision captioning disabled (no --vision_model set).")

    print(
        f"[INFO] Retrieval: multi_query_count={args.multi_query_count} | "
        f"mmr_lambda={args.mmr_lambda:.2f} | max_per_source={args.max_per_source}"
    )

    # ── Dispatch to the appropriate mode ──────────────────────────────────────

    shared = dict(
        stores=stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        llm=llm,
        args=args,
        vision_cfg=vision_cfg,
    )

    # Mode A: document error detection
    if args.document.strip():
        doc_path = Path(args.document).expanduser().resolve()
        if not doc_path.exists():
            raise SystemExit(f"Document not found: {doc_path}")
        if doc_path.suffix.lower() != ".docx":
            raise SystemExit(f"Only .docx files are supported, got: {doc_path.suffix}")
        check_document(doc_path, **shared)
        return

    # Mode B: single question
    if args.question.strip():
        answer(args.question, **shared)
        return

    # Mode C: interactive loop (original behaviour)
    print("Interactive RAG. Empty input to exit.")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            break
        answer(q, **shared)


if __name__ == "__main__":
    main()
