#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

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
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv(".env")


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and v.strip() else default


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key, "").strip()
    return int(v) if v else default

def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}

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


# ── Generic JSONL helper ──────────────────────────────────────────────────────

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


# ── Taxonomy ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ErrorCatalog:
    raw: Dict[str, Any]
    main_classes: Dict[str, Dict[str, Any]]
    sub_by_main: Dict[str, Dict[str, Dict[str, Any]]]
    sub_to_main: Dict[str, str]
    change_types: Dict[str, Dict[str, Any]]
    severity_levels: Dict[str, Dict[str, Any]]

    @property
    def allowed_main_labels(self) -> Set[str]:
        return {v["label"] for v in self.main_classes.values()}

    @property
    def allowed_subclasses_by_main_label(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for main_id, main_obj in self.main_classes.items():
            out[main_obj["label"]] = {sub["label"] for sub in self.sub_by_main.get(main_id, {}).values()}
        return out

    @property
    def allowed_change_labels(self) -> Set[str]:
        return {v["label"] for v in self.change_types.values()}

    @property
    def allowed_severity_labels(self) -> Set[str]:
        return {v["label"] for v in self.severity_levels.values()}

    @property
    def subclass_label_to_main_label(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for main_id, main_obj in self.main_classes.items():
            main_label = main_obj["label"]
            for sub in self.sub_by_main.get(main_id, {}).values():
                out[sub["label"]] = main_label
        return out


def load_taxonomy_json(path: Path) -> ErrorCatalog:
    if not path.exists():
        raise SystemExit(f"Taxonomy file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Taxonomy JSON must be a top-level object.")

    main_classes_list = raw.get("main_classes")
    change_types_list = raw.get("change_types")
    severity_levels_list = raw.get("severity_levels")

    if not isinstance(main_classes_list, list) or not isinstance(change_types_list, list) or not isinstance(severity_levels_list, list):
        raise SystemExit("Taxonomy JSON must contain main_classes, change_types, and severity_levels arrays.")

    main_classes: Dict[str, Dict[str, Any]] = {}
    sub_by_main: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sub_to_main: Dict[str, str] = {}

    for entry in main_classes_list:
        if not isinstance(entry, dict):
            continue
        main_id = str(entry.get("id") or "").strip()
        main_label = str(entry.get("label") or "").strip()
        if not main_id or not main_label:
            continue

        main_classes[main_id] = entry
        sub_by_main[main_id] = {}

        subclasses = entry.get("subclasses", [])
        if not isinstance(subclasses, list):
            continue

        for sub in subclasses:
            if not isinstance(sub, dict):
                continue
            sub_id = str(sub.get("id") or "").strip()
            sub_label = str(sub.get("label") or "").strip()
            if not sub_id or not sub_label:
                continue
            sub_by_main[main_id][sub_id] = sub
            sub_to_main[sub_label] = main_label

    change_types: Dict[str, Dict[str, Any]] = {}
    for entry in change_types_list:
        if not isinstance(entry, dict):
            continue
        _id = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or "").strip()
        if _id and label:
            change_types[_id] = entry

    severity_levels: Dict[str, Dict[str, Any]] = {}
    for entry in severity_levels_list:
        if not isinstance(entry, dict):
            continue
        _id = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or "").strip()
        if _id and label:
            severity_levels[_id] = entry

    if not main_classes or not change_types or not severity_levels:
        raise SystemExit("Taxonomy JSON contains no valid classes, change types, or severity levels.")

    return ErrorCatalog(
        raw=raw,
        main_classes=main_classes,
        sub_by_main=sub_by_main,
        sub_to_main=sub_to_main,
        change_types=change_types,
        severity_levels=severity_levels,
    )

def build_label_to_id_maps(catalog: ErrorCatalog):
    sub_label_to_id = {}
    for main_id, subs in catalog.sub_by_main.items():
        for sub_id, sub in subs.items():
            sub_label_to_id[sub["label"]] = sub_id

    change_label_to_id = {
        v["label"]: k for k, v in catalog.change_types.items()
    }

    severity_label_to_id = {
        v["label"]: k for k, v in catalog.severity_levels.items()
    }

    return sub_label_to_id, change_label_to_id, severity_label_to_id

def save_predictions_jsonl(
        report: Dict[str, Any],
        case_id: str,
        output_path: Path,
        catalog: ErrorCatalog,
) -> None:
    sub_map, change_map, severity_map = build_label_to_id_maps(catalog)

    segments: Dict[int, List[Dict[str, Any]]] = {}

    all_findings = report.get("factual_findings", []) + report.get("language_findings", [])

    for i, item in enumerate(all_findings, start=1):
        seg_idx = item.get("segment_index")
        if seg_idx is None:
            continue

        subclass_label = str(item.get("subklasse") or "").strip()
        change_label = str(item.get("aenderungstyp") or "").strip()
        severity_label = str(item.get("schweregrad") or "").strip()
        span_text = str(item.get("stelle_im_segment") or "").strip()

        if not span_text:
            continue

        subclass_id = sub_map.get(subclass_label)
        change_type_id = change_map.get(change_label)
        severity_id = severity_map.get(severity_label)

        if not subclass_id or not change_type_id or not severity_id:
            continue

        finding = {
            "finding_id": f"PRED-{case_id}-{i:04d}",
            "subclass_id": subclass_id,
            "change_type_id": change_type_id,
            "severity_id": severity_id,
            "span_text": span_text,
            "rationale": str(item.get("begruendung") or "").strip(),
        }

        vorschlag = str(item.get("vorschlag") or "").strip()
        if vorschlag:
            finding["correction"] = vorschlag

        segments.setdefault(seg_idx, []).append(finding)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for seg_idx, findings in sorted(segments.items()):
            obj = {
                "case_id": case_id,
                "segment_id": f"{case_id}_seg_{seg_idx:04d}",
                "segment_index": seg_idx,
                "predicted_findings": findings,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved predictions to {output_path}")

def build_taxonomy_block(catalog: ErrorCatalog) -> str:
    lines: List[str] = ["Zulässige Klassifikation:"]
    for main_id, main_obj in catalog.main_classes.items():
        lines.append(main_obj["label"])
        for sub in catalog.sub_by_main.get(main_id, {}).values():
            desc = str(sub.get("description") or "").strip()
            if desc:
                lines.append(f"  - {sub['label']}: {desc}")
            else:
                lines.append(f"  - {sub['label']}")

    lines.append("")
    lines.append("Zulässige Änderungstypen:")
    for obj in catalog.change_types.values():
        desc = str(obj.get("description") or "").strip()
        if desc:
            lines.append(f"  - {obj['label']}: {desc}")
        else:
            lines.append(f"  - {obj['label']}")

    lines.append("")
    lines.append("Zulässige Schweregrade:")
    for obj in catalog.severity_levels.values():
        lines.append(f"  - {obj['label']}")

    return "\n".join(lines)


# ── RAG store ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagStore:
    """
    source_kind:
      "rules"    -> global QM/QS Regelwerk
      "material" -> fallbezogene Zusatzmaterialien
    """
    name: str
    source_kind: str
    ids: np.ndarray
    emb: np.ndarray
    index_map: Dict[str, Dict[str, Any]]
    text_map: Dict[str, str]


@dataclass(frozen=True)
class Retrieved:
    rank: int
    score: float
    id: str
    meta: Dict[str, Any]
    text: str
    vec: Optional[np.ndarray] = None
    retrieval_query: str = ""


@dataclass(frozen=True)
class EvidenceSource:
    source_ref: str
    source_kind: str
    chunk_id: str
    document: str
    source_path: str
    case_id: str
    document_type: str
    chunk_index: Optional[int]
    score: float
    text: str


@dataclass(frozen=True)
class SegmentEvidence:
    segment_index: int
    segment_text: str
    retrieval_queries: List[str]
    rules_sources: List[EvidenceSource]
    material_sources: List[EvidenceSource]

    @property
    def all_sources(self) -> List[EvidenceSource]:
        return self.rules_sources + self.material_sources


# ── Artifact loaders ──────────────────────────────────────────────────────────

def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    ids = data["ids"]
    emb = data["embeddings"].astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    if len(ids) != emb.shape[0]:
        raise ValueError(f"ids length {len(ids)} != embeddings rows {emb.shape[0]}")
    return ids, emb


def load_index_map(index_path: Path) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for obj in _iter_jsonl(index_path):
        _id = obj.get("id")
        if isinstance(_id, str):
            m[_id] = obj
    return m


def load_prepared_text_map(prepared_jsonl: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for obj in _iter_jsonl(prepared_jsonl):
        _id = obj.get("id")
        txt = obj.get("text")
        if isinstance(_id, str) and isinstance(txt, str):
            m[_id] = txt
    return m


def load_rag_store(
        name: str,
        source_kind: str,
        npz_path: Path,
        index_path: Path,
        prepared_path: Path,
) -> RagStore:
    if not npz_path.exists():
        raise SystemExit(f"[{name}] Embeddings not found: {npz_path}")
    if not index_path.exists():
        raise SystemExit(f"[{name}] Index not found: {index_path}")
    if not prepared_path.exists():
        raise SystemExit(f"[{name}] Prepared JSONL not found: {prepared_path}")

    ids, emb = load_npz(npz_path)
    index_map = load_index_map(index_path)
    text_map = load_prepared_text_map(prepared_path)

    print(
        f"[INFO] Loaded RAG store '{name}': "
        f"{emb.shape[0]} chunks, dim={emb.shape[1]}, "
        f"index={len(index_map)}, texts={len(text_map)}"
    )
    return RagStore(
        name=name,
        source_kind=source_kind,
        ids=ids,
        emb=emb,
        index_map=index_map,
        text_map=text_map,
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
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = l2_normalize(pooled).to(torch.float32).cpu().numpy()
    return pooled[0].astype(np.float32)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

_STOPWORDS_DE = {
    "als", "am", "an", "auch", "aus", "bei", "bis", "dabei", "das", "dass", "dem", "den",
    "der", "des", "die", "dies", "diese", "dieser", "doch", "durch", "ein", "eine", "einem",
    "einen", "einer", "eines", "er", "es", "für", "habe", "haben", "hat", "hinter", "ich",
    "im", "in", "ist", "ja", "kann", "können", "mich", "mir", "mit", "muss", "müssen", "nach",
    "noch", "nun", "oder", "sehr", "sein", "sind", "so", "soll", "sollen", "tue", "tun", "und",
    "unter", "vom", "von", "vor", "war", "was", "welche", "welcher", "welches", "wenn", "wer",
    "wie", "wir", "wird", "wo", "zu", "zum", "zur",
}


def topk_cosine(emb: np.ndarray, qvec: np.ndarray, k: int) -> np.ndarray:
    scores = emb @ qvec
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _normalize_query_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def _extract_query_keywords(text: str, *, max_tokens: int = 8) -> str:
    raw_tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-zÄÖÜäöüß0-9][A-Za-zÄÖÜäöüß0-9\-_/]+", text)
    ]
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


def _metadata_matches(meta: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
    if not metadata_filter:
        return True
    for key, wanted in metadata_filter.items():
        if meta.get(key) != wanted:
            return False
    return True


def retrieve_from_store(
        store: RagStore,
        qvec: np.ndarray,
        k: int,
        *,
        retrieval_query: str = "",
        metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Retrieved]:
    if metadata_filter:
        filtered_rows: List[int] = []
        for i, raw_id in enumerate(store.ids):
            _id = str(raw_id)
            meta = store.index_map.get(_id, {})
            if _metadata_matches(meta, metadata_filter):
                filtered_rows.append(i)

        if not filtered_rows:
            return []

        row_idx = np.array(filtered_rows, dtype=np.int64)
        emb_sub = store.emb[row_idx]
        local_top = topk_cosine(emb_sub, qvec, k=min(k, emb_sub.shape[0]))
        idxs = row_idx[local_top]
        scores = (emb_sub @ qvec)[local_top]
    else:
        idxs = topk_cosine(store.emb, qvec, k=k)
        scores = (store.emb @ qvec)[idxs]

    hits: List[Retrieved] = []
    for i, s in zip(idxs, scores):
        _id = str(store.ids[i])
        meta = store.index_map.get(_id, {})
        text = store.text_map.get(_id, "")
        hits.append(
            Retrieved(
                rank=0,
                score=float(s),
                id=_id,
                meta=meta,
                text=text,
                vec=store.emb[i],
                retrieval_query=retrieval_query,
            )
        )
    return hits


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
        selected_ids = {s.id for s in selected}
        for cand in pool:
            if cand.id in selected_ids:
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


def _tag_hits(hits: List[Retrieved], source_kind: str) -> List[Retrieved]:
    return [
        Retrieved(
            rank=h.rank,
            score=h.score,
            id=h.id,
            meta={**h.meta, "source_kind": source_kind},
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for h in hits
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
        case_id: str = "",
        rules_top_k: int = 10,
        material_top_k: int = 10,
) -> Tuple[List[Retrieved], List[str], List[Retrieved], List[Retrieved]]:
    queries = build_multi_queries(query_text, mode=mode, max_queries=multi_query_count)

    rules_candidates: List[Retrieved] = []
    material_candidates: List[Retrieved] = []

    for query_variant in queries:
        qvec = embed_e5_query(
            query_variant,
            model=embed_model,
            tokenizer=embed_tok,
            device=device,
            max_length=max_length,
        )

        for store in stores:
            if store.source_kind == "rules":
                raw_hits = retrieve_from_store(
                    store,
                    qvec,
                    k=candidate_k,
                    retrieval_query=query_variant,
                )
                rules_candidates.extend(_tag_hits(raw_hits, "rules"))

            elif store.source_kind == "material":
                filt = {"case_id": case_id} if case_id else None
                raw_hits = retrieve_from_store(
                    store,
                    qvec,
                    k=candidate_k,
                    retrieval_query=query_variant,
                    metadata_filter=filt,
                )
                material_candidates.extend(_tag_hits(raw_hits, "case_material"))

    rules_hits = diversify_hits_mmr(
        rules_candidates,
        top_k=max(0, rules_top_k),
        mmr_lambda=mmr_lambda,
        max_per_source=max_per_source,
    )

    material_hits = diversify_hits_mmr(
        material_candidates,
        top_k=max(0, material_top_k),
        mmr_lambda=mmr_lambda,
        max_per_source=max_per_source,
    )

    final_hits = rules_hits + material_hits
    final_hits = sorted(final_hits, key=lambda h: h.score, reverse=True)

    if top_k > 0:
        final_hits = final_hits[:top_k]

    final_hits = [
        Retrieved(
            rank=i + 1,
            score=h.score,
            id=h.id,
            meta=h.meta,
            text=h.text,
            vec=h.vec,
            retrieval_query=h.retrieval_query,
        )
        for i, h in enumerate(final_hits)
    ]
    return final_hits, queries, rules_hits, material_hits


# ── Vision captioning ─────────────────────────────────────────────────────────

def enrich_hits_with_image_captions(
        hits: List[Retrieved],
        *,
        vision_cfg: dict,
        max_workers: int = 3,
) -> List[Retrieved]:
    try:
        from importDocuments_structural import ollama_caption_png
    except ImportError:
        print("[WARN] importDocuments_structural not found — skipping image captioning")
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
            rank=hit.rank,
            score=hit.score,
            id=hit.id,
            meta=hit.meta,
            text=enriched_text,
            vec=hit.vec,
            retrieval_query=hit.retrieval_query,
        )

    hits_with = [h for h in hits if h.meta.get("embedded_images")]
    hits_without = [h for h in hits if not h.meta.get("embedded_images")]
    enriched = list(hits_without)

    if hits_with:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process, h): h for h in hits_with}
            for future in as_completed(futures):
                enriched.append(future.result())
        enriched.sort(key=lambda h: h.rank)

    return enriched


# ── Context builders ──────────────────────────────────────────────────────────

def build_context_blocks(
        hits: List[Retrieved],
        *,
        max_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []
    used = 0

    for h in hits:
        m = h.meta or {}
        src = {
            "n": h.rank,
            "score": round(h.score, 4),
            "id": h.id,
            "retrieval_query": h.retrieval_query,
            "source_name": m.get("source_name") or m.get("origin_source_name"),
            "source_path": m.get("source_path") or m.get("origin_source_path"),
            "source_kind": m.get("source_kind"),
            "case_id": m.get("case_id"),
            "document_type": m.get("document_type"),
            "chunk_index": m.get("chunk_index"),
            "chunk_len": m.get("chunk_len"),
            "pdf_ocr_used": m.get("pdf_ocr_used"),
            "pdf_text_reader": m.get("pdf_text_reader"),
            "ocr_lang": m.get("ocr_lang"),
            "embedded_images": m.get("embedded_images", []),
        }
        sources.append(src)

        file_name = m.get("source_name") or m.get("origin_source_name") or "?"
        extra = []
        if m.get("chunk_index") is not None:
            extra.append(f"chunk_index={m['chunk_index']}")
        if m.get("section_title"):
            extra.append(f"section={m['section_title']!r}")
        if m.get("case_id"):
            extra.append(f"case_id={m['case_id']}")
        if m.get("document_type"):
            extra.append(f"document_type={m['document_type']}")
        extra_s = (" " + " ".join(extra)) if extra else ""

        header = (
            f"[{h.rank}] "
            f"chunk_id={h.id} "
            f"document={file_name}"
            f"{extra_s} "
            f"score={h.score:.4f}"
        )
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


def make_evidence_sources(
        hits: List[Retrieved],
        *,
        ref_prefix: str,
) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    for i, h in enumerate(hits, start=1):
        m = h.meta or {}
        sources.append(
            EvidenceSource(
                source_ref=f"{ref_prefix}_{i}",
                source_kind=str(m.get("source_kind") or ""),
                chunk_id=h.id,
                document=str(m.get("source_name") or m.get("origin_source_name") or "?"),
                source_path=str(m.get("source_path") or m.get("origin_source_path") or ""),
                case_id=str(m.get("case_id") or ""),
                document_type=str(m.get("document_type") or ""),
                chunk_index=m.get("chunk_index"),
                score=float(h.score),
                text=(h.text or "").strip(),
            )
        )
    return sources


def build_agent_context_from_sources(
        sources: List[EvidenceSource],
        *,
        max_chars: int,
) -> str:
    blocks: List[str] = []
    used = 0
    for src in sources:
        header = (
            f"{src.source_ref} | "
            f"chunk_id={src.chunk_id} | "
            f"document={src.document} | "
            f"source_kind={src.source_kind} | "
            f"chunk_index={src.chunk_index} | "
            f"score={src.score:.4f}"
        )
        body = src.text.strip()
        if not body:
            continue

        block = header + "\n" + body
        if used + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - used - len(header) - 2)
            if remaining > 0:
                blocks.append(header + "\n" + body[:remaining].rstrip() + "\n…")
            break

        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks).strip()


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
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = options
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
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
    timeout_s = env_int("LLM_TIMEOUT_S", 300)
    options = env_json_object_optional("LLM_OPTIONS_JSON")

    if backend == "ollama":
        base_url = require_env("OLLAMA_BASE_URL")
        return OllamaClient(
            base_url=base_url,
            model=model,
            options=options,
            timeout_s=timeout_s,
        )

    raise RuntimeError(f"Unsupported LLM_BACKEND: {backend!r}")


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_qa_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "Du bist ein präziser Assistent für transparente RAG-Antworten.\n"
        "Beantworte die FRAGE ausschliesslich anhand des KONTEXTS.\n"
        "Jede inhaltliche Aussage muss mit [N] belegt werden.\n"
        "Erfinde nichts und ergänze nichts aus eigenem Wissen."
    )
    user = (
        f"FRAGE:\n{question.strip()}\n\n"
        f"KONTEXT:\n{context}\n\n"
        "Antworte quellengebunden."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_factual_review_messages(
        segment_text: str,
        rules_context: str,
        material_context: str,
        catalog: ErrorCatalog,
) -> List[Dict[str, str]]:
    system = (
        "Du bist Agent 2: Fachprüfer für technische Dokumente.\n"
        "Du erhältst genau ein DOKUMENTSEGMENT, REGELWERK-QUELLEN und FALLMATERIAL-QUELLEN.\n\n"
        "Deine Aufgabe:\n"
        "Erkenne fachliche, normative, strukturelle, rechtliche oder formale Fehler im Dokumentsegment "
        "und klassifiziere sie ausschliesslich mit der vorgegebenen Taxonomie.\n\n"
        f"{build_taxonomy_block(catalog)}\n\n"
        "VERBOTEN:\n"
        "  - freie Quellenangaben ausserhalb der gegebenen source_refs\n"
        "  - erfundene source_refs\n"
        "  - erfundene Hauptklassen, Subklassen, Änderungstypen oder Schweregrade\n"
        "  - 'begruendung' darf KEINEN Text aus dem DOKUMENTSEGMENT als Beleg verwenden\n"
        "  - 'begruendung' darf NUR auf REGELWERK-QUELLEN oder FALLMATERIAL-QUELLEN basieren\n"
        "  - 'ss' als Fehler melden: In der Schweiz wird 'ss' statt 'ß' geschrieben "
        "(Fussgänger, Strasse, Fluss). Das ist KORREKT und kein Fehler.\n\n"
        "  - Fehler melden, die das Segment selbst bereits korrigiert oder erklärt\n\n"
        "PFLICHTFORMAT für 'begruendung':\n"
        "  Beginne immer mit 'Laut [SRC_X_N]:' gefolgt von der relevanten Aussage aus der Quelle.\n\n"
        "Klassifikationsregeln:\n"
        "  - Verwende pro Finding genau eine Hauptklasse und genau eine passende Subklasse.\n"
        "  - Verwende genau einen Änderungstyp.\n"
        "  - Vergib einen Schweregrad: niedrig, mittel oder hoch.\n"
        "  - Ein Finding ist nur erlaubt, wenn es direkt durch mindestens eine source_ref belegt ist.\n"
        "  - Wenn der Fehler rein sprachlich oder stilistisch ist, melde ihn NICHT hier.\n\n"
        "Wenn kein fachlich belegbarer Fehler vorliegt, antworte exakt mit: {\"errors\":[]}\n"
        "Antworte ausschliesslich als valides JSON ohne Markdown und ohne Einleitung.\n\n"
        "Format:\n"
        "{\"errors\":[{"
        "\"hauptklasse\":\"QM/QS-Konformität\","
        "\"subklasse\":\"Dokumentationspflicht\","
        "\"aenderungstyp\":\"Normativer Mangel\","
        "\"schweregrad\":\"mittel\","
        "\"stelle_im_segment\":\"<kurzer Originalausschnitt>\","
        "\"begruendung\":\"Laut [SRC_R_1]: <relevante Aussage aus der Quelle>\","
        "\"source_refs\":[\"S1_R_1\",\"S1_M_2\"]}]}"
    )
    user = (
        f"DOKUMENTSEGMENT:\n{segment_text.strip()}\n\n"
        f"REGELWERK-QUELLEN:\n{rules_context.strip()}\n\n"
        f"FALLMATERIAL-QUELLEN:\n{material_context.strip()}\n\n"
        "Erstelle die fachliche Fehlerliste als JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_language_review_messages(segment_text: str) -> List[Dict[str, str]]:
    system = (
        "Du bist Agent 3: Sprach- und Formalprüfer.\n"
        "Du meldest NUR eindeutige, lokale formale Fehler.\n\n"
        "Zulässige Taxonomie für die Ausgabe:\n"
        "  hauptklasse = Formales\n"
        "  subklasse = Redaktionelle Korrektur ODER Referenzen (formal) ODER Dokumentstruktur ODER Adressierung\n"
        "  aenderungstyp = Redaktionelle Korrektur\n"
        "  schweregrad = niedrig\n\n"
        "ERLAUBT:\n"
        "  - eindeutige Orthografiefehler\n"
        "  - eindeutige Zeichensetzungsfehler\n"
        "  - eindeutige lokale Grammatikfehler\n"
        "  - eindeutige formale Referenz- oder Adressierungsfehler\n\n"
        "NICHT ERLAUBT:\n"
        "  - Stilverbesserungen oder schönere Formulierungen\n"
        "  - minimale Umformulierungen ohne klaren Fehler\n"
        "  - Terminologieangleichungen oder 'konsistenter Gebrauch'\n"
        "  - Eigennamen, Produktnamen, Institutionen, Marken, Fahrzeugbezeichnungen, Aktenzeichen\n"
        "  - Abkürzungen, sofern nicht eindeutig falsch ausgeschrieben\n"
        "  - Korrekturen innerhalb von Anführungszeichen\n"
        "  - jede Korrektur, bei der du nicht mit sehr hoher Sicherheit sagen kannst, dass der Vorschlag korrekt ist\n"
        "  - 'ss' als Fehler melden: In der Schweiz ist 'ss' korrekt\n\n"
        "Wenn kein eindeutiger Fehler vorliegt, antworte exakt mit: {\"errors\":[]}\n"
        "Antworte ausschliesslich als valides JSON.\n\n"
        "Format:\n"
        "{\"errors\":[{"
        "\"hauptklasse\":\"Formales\","
        "\"subklasse\":\"Redaktionelle Korrektur\","
        "\"aenderungstyp\":\"Redaktionelle Korrektur\","
        "\"schweregrad\":\"niedrig\","
        "\"stelle_im_segment\":\"<exakter Originalausschnitt>\","
        "\"begruendung\":\"<warum der Fehler eindeutig ist>\","
        "\"vorschlag\":\"<nur wenn eindeutig korrekt>\"}]}"
    )
    user = (
        f"DOKUMENTSEGMENT:\n{segment_text.strip()}\n\n"
        "Melde nur eindeutige formale Fehler. "
        "Wenn du unsicher bist, gib keine Meldung aus."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def build_json_repair_messages(raw_reply: str, schema_name: str) -> List[Dict[str, str]]:
    system = (
        "Extrahiere aus der folgenden Antwort ausschliesslich ein einziges valides JSON-Objekt.\n"
        "Gib nur JSON zurück, ohne Markdown, ohne Einleitung, ohne Erklärung.\n"
        f"Schema: {schema_name}\n"
        "Falls kein verwertbares JSON vorhanden ist, gib exakt zurück: {\"errors\":[]}"
    )
    user = f"Antwort:\n{raw_reply}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── JSON parsing / normalization ──────────────────────────────────────────────

def extract_first_json_object(raw_text: str) -> str:
    s = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    decoder = json.JSONDecoder()
    first_brace = s.find("{")
    while first_brace != -1:
        try:
            obj, _end = decoder.raw_decode(s[first_brace:])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            first_brace = s.find("{", first_brace + 1)

    raise ValueError("No JSON object found in response")


def parse_json_response(raw_reply: str) -> Dict[str, Any]:
    raw_json = extract_first_json_object(raw_reply)
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")
    errors = parsed.get("errors")
    if errors is None:
        errors = parsed.get("fehler")
    if not isinstance(errors, list):
        raise ValueError("JSON must contain 'errors' or 'fehler' as a list")
    return {"errors": errors}


def normalize_factual_errors(raw_errors: List[Any], catalog: ErrorCatalog) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed_main = catalog.allowed_main_labels
    allowed_subs_by_main = catalog.allowed_subclasses_by_main_label
    allowed_change = catalog.allowed_change_labels
    allowed_severity = catalog.allowed_severity_labels

    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        hauptklasse  = str(item.get("hauptklasse")    or item.get("main_class")   or "").strip()
        subklasse    = str(item.get("subklasse")       or item.get("subcategory")  or "").strip()
        aenderungstyp= str(item.get("aenderungstyp")  or item.get("change_type")  or "").strip()
        schweregrad  = str(item.get("schweregrad")     or item.get("severity")     or "").strip()
        stelle       = str(item.get("stelle_im_segment") or item.get("stelle")     or "").strip()
        begruendung  = str(item.get("begruendung")     or item.get("begründung")   or "").strip()
        source_refs  = [str(x).strip() for x in item.get("source_refs", []) if str(x).strip()]

        if hauptklasse not in allowed_main:
            print(f"[DROP] hauptklasse {hauptklasse!r} not in {sorted(allowed_main)}")
            continue
        if subklasse not in allowed_subs_by_main.get(hauptklasse, set()):
            print(f"[DROP] subklasse {subklasse!r} not in {sorted(allowed_subs_by_main.get(hauptklasse, set()))}")
            continue
        if aenderungstyp not in allowed_change:
            print(f"[DROP] aenderungstyp {aenderungstyp!r} not in {sorted(allowed_change)}")
            continue
        if schweregrad not in allowed_severity:
            schweregrad = "mittel"
        if not source_refs:
            print(f"[DROP] no source_refs for stelle={stelle!r}")
            continue

        out.append({
            "hauptklasse":      hauptklasse,
            "subklasse":        subklasse,
            "aenderungstyp":    aenderungstyp,
            "schweregrad":      schweregrad,
            "stelle_im_segment":stelle,
            "begruendung":      begruendung,
            "source_refs":      source_refs,
        })

    return out


def normalize_language_errors(raw_errors: List[Any], catalog: ErrorCatalog) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed_subs = catalog.allowed_subclasses_by_main_label.get("Formales", set())

    legacy_to_new = {
        "Rechtschreibung": "Redaktionelle Korrektur",
        "Grammatik": "Redaktionelle Korrektur",
        "Kommafehler": "Redaktionelle Korrektur",
        "Stil": "Stil / Redundanz",
    }

    for item in raw_errors:
        if not isinstance(item, dict):
            continue

        subklasse = str(item.get("subklasse") or item.get("fehlerklasse") or "Redaktionelle Korrektur").strip()
        subklasse = legacy_to_new.get(subklasse, subklasse)

        if subklasse not in allowed_subs:
            continue

        aenderungstyp = str(item.get("aenderungstyp") or item.get("änderungstyp") or "Redaktionelle Korrektur").strip()
        if aenderungstyp not in catalog.allowed_change_labels:
            aenderungstyp = "Redaktionelle Korrektur"

        schweregrad = str(item.get("schweregrad") or item.get("fehlerschwere") or "niedrig").strip()
        if schweregrad not in catalog.allowed_severity_labels:
            schweregrad = "niedrig"

        out.append({
            "hauptklasse": "Formales",
            "subklasse": subklasse,
            "aenderungstyp": aenderungstyp,
            "schweregrad": schweregrad,
            "stelle_im_segment": str(item.get("stelle_im_segment") or item.get("stelle") or "").strip(),
            "begruendung": str(item.get("begruendung") or item.get("begründung") or "").strip(),
            "vorschlag": str(item.get("vorschlag") or "").strip(),
        })
    return out


# ── Document segmentation ─────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.!?…])(?:[\]\)\"'»”’]+)?\s+(?=(?:[A-ZÄÖÜ]|\d|[-–—•*]))"
)


def _split_long_text_on_whitespace(text: str, *, target_chars: int, max_chars: int) -> List[str]:
    s = " ".join((text or "").split())
    if not s:
        return []

    parts: List[str] = []
    remaining = s

    while len(remaining) > max_chars:
        hard_limit = min(len(remaining), max_chars)
        preferred = min(len(remaining), target_chars)

        cut = remaining.rfind(" ", 0, preferred + 1)
        if cut < max(1, preferred // 2):
            cut = remaining.rfind(" ", 0, hard_limit + 1)
        if cut == -1:
            cut = hard_limit

        part = remaining[:cut].strip()
        if part:
            parts.append(part)

        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def _split_paragraph_into_sentence_units(paragraph: str) -> List[str]:
    para = (paragraph or "").strip()
    if not para:
        return []

    if "\n" in para:
        line_units = [line.strip() for line in para.splitlines() if line.strip()]
        if len(line_units) > 1:
            units: List[str] = []
            for line in line_units:
                units.extend(_split_paragraph_into_sentence_units(line))
            return units

    if re.match(r"^(?:[-*•]\s+|\d+[\.)]\s+)", para):
        return [para]

    pieces = re.split(_SENTENCE_SPLIT_RE, para)
    units = [p.strip() for p in pieces if p and p.strip()]
    return units or [para]


def _split_paragraph_hierarchically(
        paragraph: str,
        *,
        target_chars: int,
        max_chars: int,
) -> List[str]:
    para = (paragraph or "").strip()
    if not para:
        return []

    if len(para) <= max_chars:
        return [para]

    sentence_units = _split_paragraph_into_sentence_units(para)
    if len(sentence_units) == 1 and sentence_units[0] == para:
        return _split_long_text_on_whitespace(para, target_chars=target_chars, max_chars=max_chars)

    chunks: List[str] = []
    current: List[str] = []

    def current_text() -> str:
        return " ".join(current).strip()

    def flush() -> None:
        nonlocal current
        joined = current_text()
        if joined:
            chunks.append(joined)
        current = []

    for unit in sentence_units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                flush()
            chunks.extend(_split_long_text_on_whitespace(unit, target_chars=target_chars, max_chars=max_chars))
            continue

        candidate = " ".join(current + [unit]).strip() if current else unit
        if current and len(candidate) > target_chars:
            flush()

        current.append(unit)

        if len(current_text()) >= target_chars:
            flush()

    flush()
    return chunks or [para]


def split_document_into_segments(
        document_text: str,
        *,
        target_chars: int = 1200,
        min_chars: int = 250,
        max_chars: int = 2200,
) -> List[str]:
    text = (document_text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return [text]

    atomic_units: List[str] = []
    for para in paragraphs:
        atomic_units.extend(
            _split_paragraph_hierarchically(
                para,
                target_chars=target_chars,
                max_chars=max_chars,
            )
        )

    segments: List[str] = []
    current: List[str] = []

    def current_segment() -> str:
        return "\n\n".join(current).strip()

    def flush() -> None:
        nonlocal current
        seg = current_segment()
        if seg:
            segments.append(seg)
        current = []

    for unit in atomic_units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                flush()
            segments.extend(_split_long_text_on_whitespace(unit, target_chars=target_chars, max_chars=max_chars))
            continue

        candidate = "\n\n".join(current + [unit]).strip() if current else unit
        if current and len(candidate) > target_chars:
            flush()

        current.append(unit)

        if len(current_segment()) >= target_chars:
            flush()

    flush()

    merged: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if merged and len(seg) < min_chars:
            candidate = merged[-1] + "\n\n" + seg
            if len(candidate) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(seg)

    return merged or [text]


# ── Agent 1: Evidence Builder ─────────────────────────────────────────────────

def build_segment_evidences(
        document_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
) -> Tuple[List[SegmentEvidence], List[str]]:
    segments = split_document_into_segments(document_text)
    evidences: List[SegmentEvidence] = []
    all_queries: List[str] = []

    per_segment_candidate_k = args.per_segment_candidate_k
    per_segment_rules_top_k = args.per_segment_rules_top_k
    per_segment_material_top_k = args.per_segment_material_top_k
    per_segment_total_top_k = per_segment_rules_top_k + per_segment_material_top_k

    for seg_idx, segment in enumerate(segments, start=1):
        _, queries, rules_hits, material_hits = retrieve_multi_query(
            segment,
            stores,
            embed_model=embed_model,
            embed_tok=embed_tok,
            device=device,
            max_length=args.query_max_length,
            top_k=per_segment_total_top_k,
            candidate_k=per_segment_candidate_k,
            multi_query_count=args.multi_query_count,
            mmr_lambda=args.mmr_lambda,
            max_per_source=args.max_per_source,
            mode="segment",
            case_id=args.case_id,
            rules_top_k=per_segment_rules_top_k,
            material_top_k=per_segment_material_top_k,
        )

        if vision_cfg and vision_cfg.get("vision_model"):
            rules_hits = enrich_hits_with_image_captions(
                rules_hits,
                vision_cfg=vision_cfg,
                max_workers=args.vision_workers,
            )
            material_hits = enrich_hits_with_image_captions(
                material_hits,
                vision_cfg=vision_cfg,
                max_workers=args.vision_workers,
            )

        rules_sources = make_evidence_sources(rules_hits, ref_prefix=f"S{seg_idx}_R")
        material_sources = make_evidence_sources(material_hits, ref_prefix=f"S{seg_idx}_M")

        evidences.append(
            SegmentEvidence(
                segment_index=seg_idx,
                segment_text=segment,
                retrieval_queries=queries,
                rules_sources=rules_sources,
                material_sources=material_sources,
            )
        )
        all_queries.extend(queries)

    deduped_queries = list(dict.fromkeys(q for q in all_queries if q))
    return evidences, deduped_queries


# ── Agent 2 / Agent 3 execution ───────────────────────────────────────────────

def run_factual_agent(
        llm: LLMClient,
        evidence: SegmentEvidence,
        *,
        per_agent_context_chars: int,
        catalog: ErrorCatalog,
) -> List[Dict[str, Any]]:
    rules_context = build_agent_context_from_sources(evidence.rules_sources, max_chars=per_agent_context_chars)
    material_context = build_agent_context_from_sources(evidence.material_sources, max_chars=per_agent_context_chars)

    messages = build_factual_review_messages(
        evidence.segment_text,
        rules_context,
        material_context,
        catalog,
    )
    raw_reply = llm.chat(messages)
    print(f"[DEBUG] Factual agent S{evidence.segment_index} raw reply ({len(raw_reply)} chars): {raw_reply[:200]!r}")

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        repair_messages = build_json_repair_messages(raw_reply, schema_name="factual_errors")
        repaired = llm.chat(repair_messages)
        parsed = parse_json_response(repaired)

    findings = normalize_factual_errors(parsed.get("errors", []), catalog)
    for f in findings:
        f["segment_index"] = evidence.segment_index
    return findings


def run_language_agent(llm: LLMClient, evidence: SegmentEvidence, *, catalog: ErrorCatalog) -> List[Dict[str, Any]]:
    messages = build_language_review_messages(evidence.segment_text)
    raw_reply = llm.chat(messages)

    try:
        parsed = parse_json_response(raw_reply)
    except Exception:
        repair_messages = build_json_repair_messages(raw_reply, schema_name="language_errors")
        repaired = llm.chat(repair_messages)
        parsed = parse_json_response(repaired)

    findings = normalize_language_errors(parsed.get("errors", []), catalog)
    findings = filter_language_findings_by_exact_span(findings, evidence.segment_text)
    findings = filter_language_findings_by_plausibility(findings)

    for f in findings:
        f["segment_index"] = evidence.segment_index
    return findings


def filter_language_findings_by_exact_span(
        findings: List[Dict[str, Any]],
        segment_text: str,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    normalized_segment = " ".join(segment_text.split())

    for item in findings:
        stelle = str(item.get("stelle_im_segment") or "").strip()
        if not stelle:
            continue

        normalized_stelle = " ".join(stelle.split())
        if normalized_stelle in normalized_segment:
            kept.append(item)

    return kept


def filter_language_findings_by_plausibility(
        findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []

    def _normalize_eszett(s: str) -> str:
        return s.replace("ß", "ss").lower()

    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[A-Za-zÄÖÜäöüß0-9\-]+|[^\w\s]", s, flags=re.UNICODE)

    def _looks_like_named_entity(s: str) -> bool:
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9\-]+", s)
        if not tokens:
            return False
        if len(tokens) <= 3 and any(t[:1].isupper() for t in tokens):
            return True
        if any(t.isupper() and len(t) >= 2 for t in tokens):
            return True
        return False

    def _is_minor_article_insertion(stelle: str, vorschlag: str) -> bool:
        s_tokens = _tokenize(stelle)
        v_tokens = _tokenize(vorschlag)
        added = [t for t in v_tokens if t not in s_tokens]
        return all(t.lower() in {"der", "die", "das", "dem", "den", "des", "ein", "eine", "einer", "einem", "einen"} for t in added) and len(added) <= 2

    for item in findings:
        stelle = str(item.get("stelle_im_segment") or "").strip()
        begruendung = str(item.get("begruendung") or "").strip().lower()
        vorschlag = str(item.get("vorschlag") or "").strip()

        if not stelle:
            continue
        if vorschlag and vorschlag == stelle:
            continue
        if ("punkt" in begruendung or "abschlusspunkt" in begruendung) and stelle.endswith((".", "!", "?")):
            continue
        if any(q in stelle for q in ("«", "»", "„", "“", "‚", "‘")):
            continue

        # Schweizer ss/ß
        if _normalize_eszett(stelle) == _normalize_eszett(vorschlag):
            continue

        # keine Eigennamen-/Bezeichnungs-Normalisierung
        if _looks_like_named_entity(stelle):
            continue

        # keine weichen Stilkorrekturen
        if "konsistent" in begruendung or "konsistenter gebrauch" in vorschlag.lower():
            continue

        # keine Mini-Artikel-Ergänzungen ohne klaren Fehler
        if vorschlag and _is_minor_article_insertion(stelle, vorschlag):
            continue

        # keine aggressiven Wortersetzungen bei kurzen Phrasen
        if len(stelle.split()) <= 3 and len(vorschlag.split()) <= 3:
            if stelle.lower() != vorschlag.lower():
                # nur akzeptieren, wenn Begründung klar Orthografie/Grammatik nennt
                if not any(x in begruendung for x in ["orthograf", "rechtschreib", "grammatik", "zeichensetzung", "komma"]):
                    continue

        kept.append(item)

    return kept

# ── Final Aggregator ──────────────────────────────────────────────────────────

def aggregate_reports(
        evidences: List[SegmentEvidence],
        factual_findings: List[Dict[str, Any]],
        language_findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    source_map: Dict[str, EvidenceSource] = {}
    for ev in evidences:
        for src in ev.all_sources:
            source_map[src.source_ref] = src

    aggregated_factual: List[Dict[str, Any]] = []
    for item in factual_findings:
        refs = [r for r in item.get("source_refs", []) if r in source_map]
        if not refs:
            continue

        chunk_ids: List[str] = []
        documents: List[str] = []
        source_details: List[Dict[str, Any]] = []

        seen_chunks: Set[str] = set()
        seen_docs: Set[str] = set()

        for ref in refs:
            src = source_map[ref]
            if src.chunk_id not in seen_chunks:
                chunk_ids.append(src.chunk_id)
                seen_chunks.add(src.chunk_id)
            if src.document not in seen_docs:
                documents.append(src.document)
                seen_docs.add(src.document)

            source_details.append({
                "source_ref": ref,
                "chunk_id": src.chunk_id,
                "document": src.document,
                "source_kind": src.source_kind,
                "chunk_index": src.chunk_index,
                "score": round(src.score, 4),
            })

        aggregated_factual.append({
            "segment_index": item.get("segment_index"),
            "hauptklasse": item.get("hauptklasse", ""),
            "subklasse": item.get("subklasse", ""),
            "aenderungstyp": item.get("aenderungstyp", ""),
            "schweregrad": item.get("schweregrad", ""),
            "stelle_im_segment": item.get("stelle_im_segment", ""),
            "begruendung": item.get("begruendung", ""),
            "source_refs": refs,
            "chunk_ids": chunk_ids,
            "dokumente": documents,
            "sources": source_details,
        })

    return {
        "factual_findings": aggregated_factual,
        "language_findings": language_findings,
    }


def render_combined_report(report: Dict[str, Any]) -> str:
    factual = report.get("factual_findings", []) or []
    language = report.get("language_findings", []) or []

    lines: List[str] = []

    lines.append("FACHLICHE FINDINGS")
    lines.append("-" * 90)
    if not factual:
        lines.append("Kein fachlicher Fehler gefunden.")
    else:
        for i, item in enumerate(factual, start=1):
            label = f"{item.get('hauptklasse') or 'Unklassifiziert'} > {item.get('subklasse') or 'Unklassifiziert'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            lines.append(f"   source_refs: {', '.join(item.get('source_refs') or []) or '-'}")
            lines.append(f"   chunk_id: {', '.join(item.get('chunk_ids') or []) or '-'}")
            lines.append(f"   dokument: {', '.join(item.get('dokumente') or []) or '-'}")
            lines.append("")

    lines.append("")
    lines.append("SPRACHLICHE FINDINGS")
    lines.append("-" * 90)
    if not language:
        lines.append("Kein sprachlicher Fehler gefunden.")
    else:
        for i, item in enumerate(language, start=1):
            label = f"{item.get('hauptklasse') or 'Unklassifiziert'} > {item.get('subklasse') or 'Unklassifiziert'}"
            lines.append(f"{i}. [{label}]")
            lines.append(f"   Änderungstyp: {item.get('aenderungstyp') or '-'}")
            lines.append(f"   Schweregrad: {item.get('schweregrad') or '-'}")
            lines.append(f"   Segment: {item.get('segment_index')}")
            lines.append(f"   Stelle im Segment: {item.get('stelle_im_segment') or '-'}")
            lines.append(f"   Begründung: {item.get('begruendung') or '-'}")
            if item.get("vorschlag"):
                lines.append(f"   Vorschlag: {item.get('vorschlag')}")
            lines.append("")

    return "\n".join(lines).rstrip()


# ── Shared retrieval + output helper (Q&A mode) ──────────────────────────────

def _retrieve_and_print(
        query_text: str,
        stores: List[RagStore],
        *,
        embed_model,
        embed_tok,
        device: torch.device,
        args: argparse.Namespace,
        vision_cfg: Optional[dict],
        log_key: str = "question",
) -> Tuple[str, List[Retrieved], List[str]]:
    candidate_k = max(args.top_k, min(max(args.top_k * 4, 12), 40))
    hits, queries, _, _ = retrieve_multi_query(
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
        case_id=args.case_id,
        rules_top_k=args.rules_top_k,
        material_top_k=args.material_top_k,
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


# ── Modes ─────────────────────────────────────────────────────────────────────

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
    context, _, _ = _retrieve_and_print(
        question,
        stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        args=args,
        vision_cfg=vision_cfg,
        log_key="question",
    )

    messages = build_qa_messages(question, context)
    reply = llm.chat(messages)

    print("\n" + "=" * 90)
    print("ANSWER")
    print("=" * 90)
    print(reply)


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
        catalog: ErrorCatalog,
) -> None:
    try:
        from importDocuments_structural import normalize_text, read_docx
    except ImportError as e:
        raise SystemExit(
            "importDocuments_structural.py must be in the same directory or on PYTHONPATH. "
            f"Original error: {e}"
        )

    print(f"[INFO] Reading document: {doc_path.name}")
    doc_text = normalize_text(read_docx(doc_path))

    if not doc_text.strip():
        print(f"[WARN] No text extracted from {doc_path.name} — aborting.")
        return

    print(f"[INFO] Document text: {len(doc_text)} chars")

    evidences, multi_queries = build_segment_evidences(
        doc_text,
        stores,
        embed_model=embed_model,
        embed_tok=embed_tok,
        device=device,
        args=args,
        vision_cfg=vision_cfg,
    )

    if args.print_sources:
        payload: Dict[str, Any] = {
            "document": str(doc_path),
            "document_chars": len(doc_text),
            "segments": len(evidences),
            "multi_queries": multi_queries,
            "segments_evidence": [],
        }
        for ev in evidences:
            payload["segments_evidence"].append({
                "segment_index": ev.segment_index,
                "retrieval_queries": ev.retrieval_queries,
                "rules_sources": [
                    {
                        "source_ref": s.source_ref,
                        "chunk_id": s.chunk_id,
                        "document": s.document,
                        "source_path": s.source_path,
                        "score": round(s.score, 4),
                        "source_kind": s.source_kind,
                        "case_id": s.case_id,
                        "document_type": s.document_type,
                        "chunk_index": s.chunk_index,
                    }
                    for s in ev.rules_sources
                ],
                "material_sources": [
                    {
                        "source_ref": s.source_ref,
                        "chunk_id": s.chunk_id,
                        "document": s.document,
                        "source_path": s.source_path,
                        "score": round(s.score, 4),
                        "source_kind": s.source_kind,
                        "case_id": s.case_id,
                        "document_type": s.document_type,
                        "chunk_index": s.chunk_index,
                    }
                    for s in ev.material_sources
                ],
            })
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.print_context:
        for ev in evidences:
            print("\n" + "=" * 90)
            print(f"SEGMENT {ev.segment_index}")
            print("=" * 90)
            print(ev.segment_text)

            print("\n" + "-" * 90)
            print("RULES EVIDENCE")
            print("-" * 90)
            print(build_agent_context_from_sources(
                ev.rules_sources,
                max_chars=max(1000, args.context_max_chars // 3),
            ) or "(leer)")

            print("\n" + "-" * 90)
            print("CASE MATERIAL EVIDENCE")
            print("-" * 90)
            print(build_agent_context_from_sources(
                ev.material_sources,
                max_chars=max(1000, args.context_max_chars // 3),
            ) or "(leer)")

    factual_findings: List[Dict[str, Any]] = []
    language_findings: List[Dict[str, Any]] = []

    per_agent_context_chars = max(30000, args.context_max_chars // 3)

    for ev in evidences:
        try:
            factual_findings.extend(
                run_factual_agent(
                    llm,
                    ev,
                    per_agent_context_chars=per_agent_context_chars,
                    catalog=catalog,
                )
            )
        except Exception as e:
            print(f"[WARN] Factual agent failed for segment {ev.segment_index}: {e}")

        try:
            language_findings.extend(run_language_agent(llm, ev, catalog=catalog))
        except Exception as e:
            print(f"[WARN] Language agent failed for segment {ev.segment_index}: {e}")

    report = aggregate_reports(evidences, factual_findings, language_findings)
    rendered_report = render_combined_report(report)

    if args.save_predictions_jsonl:
        save_predictions_jsonl(
            report=report,
            case_id=args.case_id,
            output_path=Path(args.save_predictions_jsonl),
            catalog=catalog,
        )

    print("\n" + "=" * 90)
    print(f"ERROR DETECTION REPORT — {doc_path.name}")
    print("=" * 90)
    print(rendered_report)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "RAG pipeline with dual-store retrieval, taxonomy loaded from JSON, and 3-agent document checking.\n\n"
            "Modes:\n"
            "  --question TEXT   Answer a free-text question\n"
            "  (no args)         Interactive Q&A loop\n"
            "  --document FILE   Detect errors in a Word document with:\n"
            "                    Agent 1 = retrieval/evidence\n"
            "                    Agent 2 = factual review\n"
            "                    Agent 3 = language review\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--question",
        type=str,
        default="",
        help="Free-text question; if empty, starts interactive loop",
    )
    mode.add_argument(
        "--document",
        type=str,
        default="",
        help="Path to a .docx file to check for errors",
    )

    ap.add_argument(
        "--case_id",
        type=str,
        default=env_str("CASE_ID", ""),
        help="Case ID for filtering case materials, e.g. case_01",
    )
    ap.add_argument(
        "--taxonomy_json",
        type=str,
        default=env_str("TAXONOMY_JSON", "taxonomy.json"),
        help="Path to taxonomy JSON",
    )

    ap.add_argument("--embeddings", type=str,
                    default=env_str("EMBED_OUT_NPZ", "embeddings.npz"),
                    help="Embeddings .npz for RAG store 1 (rules)")
    ap.add_argument("--index", type=str,
                    default=env_str("EMBED_OUT_INDEX", "index.jsonl"),
                    help="Index .jsonl for RAG store 1")
    ap.add_argument("--prepared", type=str,
                    default=env_str("OUT_JSONL", "prepared.jsonl"),
                    help="Prepared .jsonl for RAG store 1")

    ap.add_argument("--embeddings2", type=str,
                    default=env_str("EMBED_OUT_NPZ2", ""),
                    help="Embeddings .npz for RAG store 2 (materials)")
    ap.add_argument("--index2", type=str,
                    default=env_str("EMBED_OUT_INDEX2", ""),
                    help="Index .jsonl for RAG store 2")
    ap.add_argument("--prepared2", type=str,
                    default=env_str("OUT_JSONL2", ""),
                    help="Prepared .jsonl for RAG store 2")

    ap.add_argument("--top_k", type=int,
                    default=env_int("TOP_K", 12),
                    help="Overall retrieval top-k")
    ap.add_argument("--rules_top_k", type=int,
                    default=env_int("RULES_TOP_K", 8),
                    help="Final number of rule chunks to keep")
    ap.add_argument("--material_top_k", type=int,
                    default=env_int("MATERIAL_TOP_K", 8),
                    help="Final number of material chunks to keep")

    ap.add_argument("--context_max_chars", type=int,
                    default=env_int("CONTEXT_MAX_CHARS", 12000),
                    help="Maximum characters used for contexts")

    ap.add_argument("--embed_model", type=str,
                    default=env_str("EMBED_MODEL", "intfloat/multilingual-e5-large"))
    ap.add_argument("--embed_device", type=str,
                    default=env_str("EMBED_DEVICE", "auto"))
    ap.add_argument("--query_max_length", type=int,
                    default=env_int("QUERY_MAX_LENGTH", 256))
    ap.add_argument("--multi_query_count", type=int,
                    default=env_int("MULTI_QUERY_COUNT", 4),
                    help="Number of internal retrieval query variants")
    ap.add_argument("--mmr_lambda", type=float,
                    default=float(env_str("MMR_LAMBDA", "0.75")),
                    help="MMR relevance weight between 0 and 1")
    ap.add_argument("--max_per_source", type=int,
                    default=env_int("MAX_PER_SOURCE", 2),
                    help="Maximum number of final chunks per source file")

    ap.add_argument("--vision_model", type=str,
                    default=env_str("VISION_MODEL", ""),
                    help="Ollama vision model, e.g. qwen2.5vl:7b. Leave empty to skip.")
    ap.add_argument("--vision_workers", type=int,
                    default=env_int("VISION_WORKERS", 3),
                    help="Parallel workers for image captioning")
    ap.add_argument("--vision_timeout_s", type=int,
                    default=env_int("VISION_TIMEOUT_S", 180))

    ap.add_argument("--print_sources", action="store_true",
                    help="Print sources metadata as JSON")
    ap.add_argument("--print_context", action="store_true",
                    help="Print segment texts and evidence blocks")
    ap.add_argument(
    "--save_predictions_jsonl",
        type=str,
        default="",
        help="Path to save structured predictions JSONL"
    )
    ap.add_argument(
        "--per_segment_candidate_k",
        type=int,
        default=env_int("PER_SEGMENT_CANDIDATE_K", ""),
        help="Candidate pool size per document segment",
    )

    ap.add_argument(
        "--per_segment_rules_top_k",
        type=int,
        default=env_int("PER_SEGMENT_RULES_TOP_K", ""),
        help="Final number of rule chunks kept per segment",
    )

    ap.add_argument(
        "--per_segment_material_top_k",
        type=int,
        default=env_int("PER_SEGMENT_MATERIAL_TOP_K", ""),
        help="Final number of material chunks kept per segment",
    )

    return ap.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    started_at = datetime.now()
    t0 = time.perf_counter()
    print(f"[INFO] Script start: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        args = parse_args()
        catalog = load_taxonomy_json(Path(args.taxonomy_json).resolve())
        print(
            f"[INFO] Loaded taxonomy: {len(catalog.main_classes)} Hauptklassen | "
            f"{sum(len(v) for v in catalog.sub_by_main.values())} Subklassen | "
            f"{len(catalog.change_types)} Änderungstypen"
        )

        stores: List[RagStore] = [
            load_rag_store(
                "rules",
                "rules",
                npz_path=Path(args.embeddings).resolve(),
                index_path=Path(args.index).resolve(),
                prepared_path=Path(args.prepared).resolve(),
            )
        ]

        if args.embeddings2.strip():
            stores.append(
                load_rag_store(
                    "material",
                    "material",
                    npz_path=Path(args.embeddings2).resolve(),
                    index_path=Path(args.index2).resolve(),
                    prepared_path=Path(args.prepared2).resolve(),
                )
            )
        else:
            print("[INFO] No second RAG store configured — using rules store only.")

        device = choose_device(args.embed_device)
        print(f"[INFO] Embedding device: {device} | model: {args.embed_model}")
        embed_model, embed_tok = load_hf_model(args.embed_model, device)

        llm = make_llm_client()

        vision_cfg: Optional[dict] = None
        vision_model_enabled = env_bool("VISION_MODEL_ENABLED", True)


        if args.vision_model.strip() and vision_model_enabled:
            vision_cfg = {
                "vision_model": args.vision_model,
                "vision_model_enabled": vision_model_enabled,
                "vision_prompt": env_str("VISION_PROMPT", ""),
                "ollama_base_url": require_env("OLLAMA_BASE_URL"),
                "vision_timeout_s": args.vision_timeout_s,
                "vision_options": env_json_object_optional("VISION_OPTIONS_JSON"),
            }
            print(f"[INFO] Vision captioning enabled: {args.vision_model}")
        else:
            print(
                f"[INFO] Vision captioning disabled: "
                f"vision_model={args.vision_model!r}, vision_model_enabled={vision_model_enabled}"
            )

        print(
            f"[INFO] Retrieval: multi_query_count={args.multi_query_count} | "
            f"mmr_lambda={args.mmr_lambda:.2f} | max_per_source={args.max_per_source} | "
            f"rules_top_k={args.rules_top_k} | material_top_k={args.material_top_k}"
        )

        shared = dict(
            stores=stores,
            embed_model=embed_model,
            embed_tok=embed_tok,
            device=device,
            llm=llm,
            args=args,
            vision_cfg=vision_cfg,
            catalog=catalog,
        )

        if args.document.strip():
            doc_path = Path(args.document).expanduser().resolve()
            if not doc_path.exists():
                raise SystemExit(f"Document not found: {doc_path}")
            if doc_path.suffix.lower() != ".docx":
                raise SystemExit(f"Only .docx files are supported, got: {doc_path.suffix}")
            if args.embeddings2.strip() and not args.case_id.strip():
                raise SystemExit("--case_id is required in document mode when using case materials.")
            check_document(doc_path, **shared)
            return

        if args.question.strip():
            answer(args.question, stores=stores, embed_model=embed_model, embed_tok=embed_tok, device=device, llm=llm, args=args, vision_cfg=vision_cfg)
            return

        print("Interactive RAG. Empty input to exit.")
        while True:
            q = input("\nQuestion> ").strip()
            if not q:
                break
            answer(q, stores=stores, embed_model=embed_model, embed_tok=embed_tok, device=device, llm=llm, args=args, vision_cfg=vision_cfg)

    finally:
        ended_at = datetime.now()
        elapsed_s = time.perf_counter() - t0
        elapsed_min = elapsed_s / 60.0
        elapsed_h = elapsed_s / 3600.0
        print(f"[INFO] Script end:   {ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INFO] Total runtime: {elapsed_s:.1f} s | {elapsed_min:.1f} min | {elapsed_h:.2f} h")


if __name__ == "__main__":
    main()
