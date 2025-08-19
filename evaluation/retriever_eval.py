import json
from pathlib import Path
from typing import List
from typing import Dict, Tuple, Optional
import os
import re
import urllib.parse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import chromadb

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)  # loads .env from project root if present

GOLD_PROMPTS_PATH = "evaluation/gold_prompts.json"
DB_PATH = "db/minecraft_lore"
COLLECTION_NAME = "minecraft_lore"
K_DEFAULT = 5  # default top-k for retrieval


def _load_gold_prompts(path: Path) -> List[dict]:
    """
    Load and lightly validate the gold prompts JSON file.

    Args:
        path (Path): Path to a JSON file containing a list of prompt dicts.

    Return:
        List[dict]: Parsed list of gold prompt items.
    """
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError("gold_prompts.json must be a JSON array of prompt objects.")
    required_keys = ["expected_answer_contains", "question", "source_link", "comment"]
    for prompt in prompts:
        for key in required_keys:
            if key not in prompt or prompt[key] in (None, "", []):
                print(f"❌ Warning: Missing or empty '{key}' in prompt: {prompt}")
    print(f"✅ Loaded {len(prompts)} gold prompts from {path}")
    return prompts


def _normalize_url(url: Optional[str]) -> str:
    """
    Normalize a URL for robust equality checks.

    Args:
        url (str | None): URL string to normalize.

    Return:
        str: Normalized URL without scheme, fragments, query params, or trailing slash.
    """
    if not url:
        return ""
    url = url.strip()
    try:
        parts = urllib.parse.urlsplit(url)
        netloc = parts.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parts.path.rstrip("/")
        # ignore query and fragment by design
        return f"{netloc}{path}"
    except Exception:
        # Fallback: lowercased, stripped scheme and trailing slash
        url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        url = url.rstrip("/").lower()
        url = url[4:] if url.startswith("www.") else url
        return url


def _normalize_text(text: Optional[str]) -> str:
    """
    Normalize free text for substring matching.

    Args:
        text (str | None): Text to normalize.

    Return:
        str: Lowercased, whitespace-collapsed text.
    """
    if not text:
        return ""
    lowered = text.lower()
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


def _extract_url_from_hit(hit) -> str:
    """
    Extract a URL from a retrieved hit's metadata or text as a fallback.

    Args:
        hit (Any): LlamaIndex NodeWithScore-like object.

    Return:
        str: A URL string if found, else empty string.
    """
    md = hit.metadata or {}
    # Common metadata keys we might have used during indexing
    for key in ("url", "source_url", "page_url", "origin_url", "link"):
        val = md.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Sometimes file-based loaders store paths
    file_path = md.get("file_path")
    if isinstance(file_path, str) and file_path.startswith("http"):
        return file_path
    # Fallback: try to find a minecraft wiki link in the chunk text
    text = hit.get_text() or ""
    m = re.search(r"https?://(?:www\.)?minecraft\.wiki[^\s\)]*", text)
    return m.group(0) if m else ""


def _get_retriever(k_default: int = K_DEFAULT):
    """
    Create and return a LlamaIndex retriever backed by Chroma.

    Args:
        k_default (int): Default top-k for similarity search.

    Return:
        Any: A LlamaIndex retriever object with similarity_top_k preset.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    try:
        print(
            f"[Chroma] Collection '{COLLECTION_NAME}' contains {collection.count()} items at '{DB_PATH}'"
        )
    except Exception:
        pass
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Ensure the same embedding model used during indexing
    embed_model_name = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    embed_model = OpenAIEmbedding(model=embed_model_name)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    retriever = index.as_retriever(similarity_top_k=k_default)
    print(
        f"✅ Retriever initialized with collection '{COLLECTION_NAME}' containing {collection.count()} items"
    )
    return retriever


def _evaluate_retriever(
    gold_prompts: List[dict], retriever, k: int = K_DEFAULT
) -> Tuple[List[dict], Dict[str, float]]:
    """
    Evaluate retrieval quality against gold prompts without invoking an LLM.

    Args:
        gold_prompts (List[dict]): List of evaluation items loaded from JSON.
        retriever: LlamaIndex retriever instance (similarity_top_k is used).
        k (int): Cutoff K for top-k retrieval.

    Return:
        Tuple[List[dict], Dict[str, float]]: (per-row results, summary metrics).
    """
    if hasattr(retriever, "similarity_top_k"):
        setattr(retriever, "similarity_top_k", k)

    rows: List[dict] = []
    hit_url_sum = 0.0
    mrr_sum = 0.0
    contains_all_sum = 0.0
    n = len(gold_prompts)

    for i, item in enumerate(gold_prompts, start=1):
        question: str = item.get("question", "")
        expected_snippets: List[str] = item.get("expected_answer_contains", []) or []
        expected_url_raw: str = item.get("source_link", "") or ""

        expected_url = _normalize_url(expected_url_raw)
        expected_norm_snips = [_normalize_text(s) for s in expected_snippets if s]

        hits = retriever.retrieve(question)

        # Build normalized views of retrieved data
        hit_texts_norm = [_normalize_text(h.get_text()) for h in hits]
        hit_urls_norm = [_normalize_url(_extract_url_from_hit(h)) for h in hits]
        hit_titles = [str((h.metadata or {}).get("source") or "") for h in hits]
        hit_scores = [float(getattr(h, "score", 0.0)) for h in hits]

        for rank, hit in enumerate(hits, start=1):
            print(
                f"      Hit {rank}: URL={_extract_url_from_hit(hit)}, Score={getattr(hit, 'score', 0.0)}"
            )

        # URL Hit@K and MRR@K (Mean Reciprocal Rank)
        hit_at_k_url = 0.0  # how often does the retriever bring back the right page within K results
        mrr_at_k_url = 0.0  # how high, on average, the right page is ranked.
        if expected_url:
            for rank, u in enumerate(hit_urls_norm, start=1):
                if u and u == expected_url:
                    hit_at_k_url = 1.0
                    mrr_at_k_url = 1.0 / rank
                    break

        # ContainsAll@K: each snippet must appear in at least one of the top-k chunks (not necessarily the same chunk)
        def snippet_covered(snippet_norm: str) -> bool:
            return any(snippet_norm in txt for txt in hit_texts_norm)

        contains_all_at_k = (
            1.0
            if expected_norm_snips
            and all(snippet_covered(s) for s in expected_norm_snips)
            else 0.0
        )

        # Row
        rows.append(
            {
                "question": question,
                "expected_answer_contains": expected_snippets,
                "source_link": expected_url_raw,
                "k": k,
                "hit_at_k_url": hit_at_k_url if expected_url else None,
                "mrr_at_k_url": mrr_at_k_url if expected_url else None,
                "contains_all_at_k": contains_all_at_k,
                "top1_url": hit_urls_norm[0] if hit_urls_norm else "",
                "top1_score": hit_scores[0] if hit_scores else None,
                "topk_urls": hit_urls_norm,
                "topk_titles": hit_titles,
                "comment": item.get("comment", ""),
            }
        )

        if expected_url:
            hit_url_sum += hit_at_k_url
            mrr_sum += mrr_at_k_url
        contains_all_sum += contains_all_at_k

        status = "✔" if (contains_all_at_k == 1.0 or hit_at_k_url == 1.0) else "✘"
        print(f"[{i}/{n}] {status}  {question}")

    denom_url = sum(1 for it in gold_prompts if (it.get("source_link") or "").strip())
    summary: Dict[str, float] = {
        "k": float(k),
        "hit_at_k_url": (hit_url_sum / denom_url) if denom_url else 0.0,
        "mrr_at_k_url": (mrr_sum / denom_url) if denom_url else 0.0,
        "contains_all_at_k": contains_all_sum / n if n else 0.0,
    }

    print("\nSummary")
    print(f"  Hit@{k} (URL):       {summary['hit_at_k_url']:.3f}")
    print(f"  MRR@{k} (URL):       {summary['mrr_at_k_url']:.3f}")
    print(f"  ContainsAll@{k}:     {summary['contains_all_at_k']:.3f}")

    return rows, summary


def main() -> None:
    print("➡️ Starting retriever evaluation ...")
    gold_prompts = _load_gold_prompts(Path(GOLD_PROMPTS_PATH))
    retriever = _get_retriever(k_default=K_DEFAULT)
    rows, summary = _evaluate_retriever(gold_prompts, retriever, k=K_DEFAULT)

    # Persist results for inspection
    out_dir = Path("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "retriever_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "retriever_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("✅ Evaluation completed")


if __name__ == "__main__":
    main()
