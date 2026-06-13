"""
Embedding Service

Handles text embedding generation via remote endpoint with caching.
"""

import hashlib
import json
import os
import random
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import load_config as _load_config
from ..validation import serialize_f32, l2_normalize

# Force-reload on module (re-)import so that tests patching os.environ and then
# reloading kb.core.embedding get fresh config values rather than a stale singleton.
_load_config(force_reload=True)


class EmbeddingService:
    """Manages text embeddings with LRU caching.

    Embeddings are L2-normalized so L2 distance can be used for cosine similarity.
    For normalized vectors: cosine_similarity = 1 - L2_distance²/2

    Supports two request/response formats:
      - llamacpp: {"content": text} request, per-token embedding arrays needing
        mean-pooling. Default for back-compat with ash:8081.
      - openai: {"input": text, "model": M} request, single pre-pooled embedding
        at data[0].embedding. Covers OpenAI, Voyage, Jina, Ollama, TEI etc.
        Sends Authorization: Bearer header when embedding_key is set.
    """

    _cache: dict[str, list[float]]
    _cache_order: list[str]
    _cache_max: int
    embedding_url: str
    embedding_dim: int
    embedding_format: str
    embedding_model: str
    embedding_key: str

    def __init__(
        self,
        embedding_url: str | None = None,
        embedding_dim: int | None = None,
        cache_max: int = 500,
        embedding_format: str | None = None,
        embedding_model: str | None = None,
        embedding_key: str | None = None,
    ):
        cfg = _load_config()
        self.embedding_url = embedding_url if embedding_url is not None else cfg.embedding_url
        self.embedding_dim = embedding_dim if embedding_dim is not None else cfg.embedding_dim
        self._cache_max = cache_max
        self._cache = {}
        self._cache_order = []
        self.embedding_format = embedding_format if embedding_format is not None else cfg.embedding_format
        self.embedding_model = embedding_model if embedding_model is not None else cfg.embedding_model
        self.embedding_key = embedding_key if embedding_key is not None else cfg.embedding_key

    def _cache_get(self, text_hash: str) -> list[float] | None:
        """Get embedding from cache, updating LRU order."""
        if text_hash in self._cache:
            self._cache_order.remove(text_hash)
            self._cache_order.append(text_hash)
            return self._cache[text_hash]
        return None

    def _cache_put(self, text_hash: str, embedding: list[float]) -> None:
        """Add embedding to cache with LRU eviction."""
        if text_hash in self._cache:
            self._cache_order.remove(text_hash)
            self._cache_order.append(text_hash)
            return
        if len(self._cache) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[text_hash] = embedding
        self._cache_order.append(text_hash)

    def _embed_remote_llamacpp(self, text: str, timeout: float) -> list[float]:
        """Fetch embedding using llama.cpp format: {"content": text}.

        Response is per-token arrays; mean-pools to a single vector.
        """
        req = Request(
            self.embedding_url,
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # llama.cpp format: [{"index": 0, "embedding": [[tok1], [tok2], ...]}]
            token_embeddings = data[0]["embedding"]
            if len(token_embeddings) == 1:
                return list(token_embeddings[0])
            # Mean pooling across all token embeddings
            dim = len(token_embeddings[0])
            pooled = [0.0] * dim
            for tok_emb in token_embeddings:
                for i, v in enumerate(tok_emb):
                    pooled[i] += v
            n = len(token_embeddings)
            return [v / n for v in pooled]

    def _embed_remote_openai(self, text: str, timeout: float) -> list[float]:
        """Fetch embedding using OpenAI-compatible format.

        POSTs {"input": text, "model": M, "dimensions": D} to self.embedding_url.
        The `dimensions` field is omitted on retry if the provider returns HTTP 400
        mentioning "dimensions" (some providers reject it).

        Response: data[0].embedding — already a single pooled vector; NO mean-pooling.
        Adds Authorization: Bearer header when self.embedding_key is set.
        """
        headers = {"Content-Type": "application/json"}
        if self.embedding_key:
            headers["Authorization"] = f"Bearer {self.embedding_key}"

        body: dict = {"input": text}
        if self.embedding_model:
            body["model"] = self.embedding_model
        body["dimensions"] = self.embedding_dim

        req = Request(
            self.embedding_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return list(data["data"][0]["embedding"])
        except HTTPError as e:
            # On HTTP 400 mentioning "dimensions" in the response body, retry
            # without that field (some providers reject it).
            if e.code == 400:
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
                if "dimension" in err_body.lower():
                    retry_body = {k: v for k, v in body.items() if k != "dimensions"}
                    req2 = Request(
                        self.embedding_url,
                        data=json.dumps(retry_body).encode("utf-8"),
                        headers=headers,
                    )
                    with urlopen(req2, timeout=timeout) as resp2:
                        data2 = json.loads(resp2.read().decode("utf-8"))
                        return list(data2["data"][0]["embedding"])
            raise

    def _embed_remote(
        self, text: str, max_retries: int | None = None, base_delay: float = 1.5,
        timeout: float | None = None,
    ) -> list[float]:
        """Get embedding from remote endpoint.

        Dispatches to the format-specific helper (_embed_remote_llamacpp or
        _embed_remote_openai) based on self.embedding_format.

        Retries with exponential backoff + jitter on failure. No fallback to local
        model to prevent dimension mismatch errors.

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts. If None, uses
                KB_EMBED_MAX_RETRIES env var (default 5). Interactive `kb add`
                sets this to 1 for fast-fail-and-queue behavior; flush-pending
                and reembed leave it at the default for their longer retry budget.
            base_delay: Base delay in seconds (doubles each retry with jitter)

        Raises:
            RuntimeError: If all retries fail
        """
        if max_retries is None:
            max_retries = int(os.environ.get("KB_EMBED_MAX_RETRIES", "5"))
        last_error: Exception | None = None
        _to = float(timeout) if timeout else float(os.environ.get("KB_EMBED_TIMEOUT", "180"))

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, delay * 0.25)
                delay += jitter
                print(f"Embedding retry {attempt}/{max_retries} after {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)

            try:
                if self.embedding_format == "openai":
                    return self._embed_remote_openai(text, _to)
                else:
                    return self._embed_remote_llamacpp(text, _to)
            except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError,
                    ConnectionError, OSError) as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Remote embedding failed after {max_retries} retries: {last_error}. "
            + f"Check that embedding server at {self.embedding_url} is running."
        )

    def embed(self, text: str, max_retries: int | None = None,
              timeout: float | None = None) -> bytes:
        """Generate embedding for text using remote endpoint.

        Embeddings are L2-normalized so L2 distance can be used for cosine similarity.
        For normalized vectors: cosine_similarity = 1 - L2_distance²/2

        Results are cached (LRU, max entries configurable) to avoid redundant API calls.

        Args:
            max_retries: retry budget passed to _embed_remote. None uses the
                KB_EMBED_MAX_RETRIES default (5). The interactive SEARCH path
                passes a small value so a down embedding server fails fast and
                degrades to FTS instead of blocking ~46s on exponential backoff.

        Raises:
            RuntimeError: If embedding_url is not configured, or all retries fail
        """
        if not self.embedding_url:
            raise RuntimeError(
                "KB_EMBEDDING_URL not configured. Set this environment variable to your embedding endpoint."
            )

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = self._cache_get(text_hash)
        if cached is not None:
            return serialize_f32(cached)

        embedding = self._embed_remote(text, max_retries=max_retries, timeout=timeout)
        embedding = l2_normalize(embedding)
        self._cache_put(text_hash, embedding)
        return serialize_f32(embedding)

    def embed_raw(self, text: str) -> list[float]:
        """Generate embedding and return as list of floats (not serialized)."""
        if not self.embedding_url:
            raise RuntimeError(
                "KB_EMBEDDING_URL not configured. Set this environment variable to your embedding endpoint."
            )

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = self._cache_get(text_hash)
        if cached is not None:
            return cached

        embedding = self._embed_remote(text)
        embedding = l2_normalize(embedding)
        self._cache_put(text_hash, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[bytes]:
        """Generate embeddings for multiple texts in one request (batch API).

        Returns list of serialized embeddings (same order as input).
        Falls back to single embed if batch request fails.
        """
        if not self.embedding_url:
            raise RuntimeError("KB_EMBEDDING_URL not configured.")

        # Check cache first
        hashes = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts]
        results: list[bytes | None] = [None] * len(texts)
        uncached_indices = []
        for i, h in enumerate(hashes):
            cached = self._cache_get(h)
            if cached is not None:
                results[i] = serialize_f32(cached)
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results  # type: ignore

        uncached_texts = [texts[i] for i in uncached_indices]
        _batch_timeout = int(os.environ.get("KB_EMBED_BATCH_TIMEOUT", "300"))

        try:
            if self.embedding_format == "openai":
                headers = {"Content-Type": "application/json"}
                if self.embedding_key:
                    headers["Authorization"] = f"Bearer {self.embedding_key}"
                body: dict = {"input": uncached_texts}
                if self.embedding_model:
                    body["model"] = self.embedding_model
                req = Request(
                    self.embedding_url,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                )
                with urlopen(req, timeout=_batch_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    # OpenAI batch format: {"data": [{"index": N, "embedding": [...]}]}
                    # Sort by index to match input order
                    items = sorted(data["data"], key=lambda x: x["index"])
                    for idx, item in enumerate(items):
                        orig_i = uncached_indices[idx]
                        vec = l2_normalize(list(item["embedding"]))
                        self._cache_put(hashes[orig_i], vec)
                        results[orig_i] = serialize_f32(vec)
            else:
                req = Request(
                    self.embedding_url,
                    data=json.dumps({"content": uncached_texts}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=_batch_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    # llama.cpp batch format: list of [{"index": N, "embedding": [[...]]}]
                    for idx, item in enumerate(data):
                        orig_i = uncached_indices[idx]
                        token_embeddings = item["embedding"]
                        if len(token_embeddings) == 1:
                            vec = list(token_embeddings[0])
                        else:
                            dim = len(token_embeddings[0])
                            pooled = [0.0] * dim
                            for tok_emb in token_embeddings:
                                for j, v in enumerate(tok_emb):
                                    pooled[j] += v
                            vec = [v / len(token_embeddings) for v in pooled]
                        vec = l2_normalize(vec)
                        self._cache_put(hashes[orig_i], vec)
                        results[orig_i] = serialize_f32(vec)
        except Exception:
            # Fallback: embed one at a time
            for i in uncached_indices:
                results[i] = self.embed(texts[i])

        return results  # type: ignore
