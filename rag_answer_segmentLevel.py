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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def topk_cosine(emb: np.ndarray, qvec: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k rows by cosine similarity (dot product, normalised)."""
    scores = emb @ qvec
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def retrieve_from_store(
        store: RagStore,
        qvec: np.ndarray,
        k: int,
) -> List[Retrieved]:
    """Query a single RAG store and return top-k hits (unranked — rank assigned later)."""
    idxs   = topk_cosine(store.emb, qvec, k=k)
    scores = (store.emb @ qvec)[idxs]
    hits: List[Retrieved] = []
    for i, s in zip(idxs, scores):
        _id  = str(store.ids[i])
        meta = store.index_map.get(_id, {})
        text = store.text_map.get(_id, "")
        hits.append(Retrieved(rank=0, score=float(s), id=_id, meta=meta, text=text))
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
        Retrieved(rank=i + 1, score=h.score, id=h.id, meta=h.meta, text=h.text)
        for i, h in enumerate(merged)
    ]


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
) -> Tuple[str, List[Retrieved], List[str]]:
    """
    Segment-level retrieval for document checking.

    Each segment is embedded and queried separately. Hits are deduplicated across
    segments, globally re-ranked by score, optionally vision-enriched, and then
    packed into one final context for the document-level checker.
    """
    segments = split_document_into_segments(document_text)
    all_hits: List[Retrieved] = []

    for segment in segments:
        qvec = embed_e5_query(
            segment,
            model=embed_model,
            tokenizer=embed_tok,
            device=device,
            max_length=args.query_max_length,
        )
        hits_per_store = [retrieve_from_store(s, qvec, k=per_segment_k) for s in stores]
        merged = merge_hits(hits_per_store, top_k=per_segment_k)
        all_hits.extend(merged)

    # Global dedupe and rerank across all segment queries
    by_id: Dict[str, Retrieved] = {}
    for hit in all_hits:
        prev = by_id.get(hit.id)
        if prev is None or hit.score > prev.score:
            by_id[hit.id] = hit

    ranked = sorted(by_id.values(), key=lambda h: h.score, reverse=True)[:args.top_k]
    ranked = [
        Retrieved(rank=i + 1, score=h.score, id=h.id, meta=h.meta, text=h.text)
        for i, h in enumerate(ranked)
    ]

    if vision_cfg and vision_cfg.get("vision_model"):
        ranked = enrich_hits_with_image_captions(
            ranked,
            vision_cfg=vision_cfg,
            max_workers=args.vision_workers,
        )

    context, _sources = build_context_blocks(ranked, max_chars=args.context_max_chars)
    return context, ranked, segments


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
) -> Tuple[str, List[Retrieved]]:
    """
    Shared retrieval pipeline used by both Q&A and document checking:
      1. Embed the query text
      2. Retrieve from all stores and merge
      3. Optionally enrich with vision captions
      4. Build and optionally print context + sources

    Returns (context_str, ranked_hits) for the caller to use in prompting.
    """
    qvec = embed_e5_query(
        query_text,
        model=embed_model,
        tokenizer=embed_tok,
        device=device,
        max_length=args.query_max_length,
    )

    all_hits = [retrieve_from_store(s, qvec, k=args.top_k) for s in stores]
    hits     = merge_hits(all_hits, top_k=args.top_k)

    if vision_cfg and vision_cfg.get("vision_model"):
        hits = enrich_hits_with_image_captions(
            hits,
            vision_cfg=vision_cfg,
            max_workers=args.vision_workers,
        )

    context, sources = build_context_blocks(hits, max_chars=args.context_max_chars)

    if args.print_sources:
        print(json.dumps({log_key: query_text[:120], "sources": sources},
                         ensure_ascii=False, indent=2))

    if args.print_context:
        print("\n" + "=" * 90)
        print("RETRIEVED CONTEXT")
        print("=" * 90)
        print(context)

    return context, hits


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
    context, _ = _retrieve_and_print(
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
    context, hits, segments = retrieve_document_context_by_segments(
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