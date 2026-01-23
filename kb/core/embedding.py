"""
Embedding Service

Handles text embedding generation via remote endpoint with caching.
"""

import hashlib
import json
import random
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..constants import DEFAULT_EMBEDDING_URL, DEFAULT_EMBEDDING_DIM
from ..validation import serialize_f32, l2_normalize


class EmbeddingService:
    """Manages text embeddings with LRU caching.

    Embeddings are L2-normalized so L2 distance can be used for cosine similarity.
    For normalized vectors: cosine_similarity = 1 - L2_distance²/2
    """

    _cache: dict[str, list[float]]
    _cache_order: list[str]
    _cache_max: int
    embedding_url: str
    embedding_dim: int

    def __init__(
        self,
        embedding_url: str = DEFAULT_EMBEDDING_URL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        cache_max: int = 500,
    ):
        self.embedding_url = embedding_url
        self.embedding_dim = embedding_dim
        self._cache_max = cache_max
        self._cache = {}
        self._cache_order = []

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

    def _embed_remote(
        self, text: str, max_retries: int = 5, base_delay: float = 2.0
    ) -> list[float]:
        """Get embedding from remote endpoint (llama.cpp style).

        llama.cpp returns per-token embeddings. We use mean pooling to get
        a single embedding for the entire text.

        Retries with exponential backoff + jitter on failure. No fallback to local
        model to prevent dimension mismatch errors.

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts (default 5 for overloaded servers)
            base_delay: Base delay in seconds (doubles each retry with jitter)

        Raises:
            RuntimeError: If all retries fail
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, delay * 0.25)
                delay += jitter
                print(f"Embedding retry {attempt}/{max_retries} after {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)

            req = Request(
                self.embedding_url,
                data=json.dumps({"content": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    # llama.cpp format: [{"index": 0, "embedding": [[tok1], [tok2], ...]}]
                    # Mean pool across all token embeddings
                    token_embeddings = data[0]["embedding"]
                    if len(token_embeddings) == 1:
                        return list(token_embeddings[0])
                    # Mean pooling
                    dim = len(token_embeddings[0])
                    pooled = [0.0] * dim
                    for tok_emb in token_embeddings:
                        for i, v in enumerate(tok_emb):
                            pooled[i] += v
                    n = len(token_embeddings)
                    return [v / n for v in pooled]
            except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError,
                    ConnectionError, OSError) as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Remote embedding failed after {max_retries} retries: {last_error}. "
            + f"Check that embedding server at {self.embedding_url} is running."
        )

    def embed(self, text: str) -> bytes:
        """Generate embedding for text using remote endpoint.

        Embeddings are L2-normalized so L2 distance can be used for cosine similarity.
        For normalized vectors: cosine_similarity = 1 - L2_distance²/2

        Results are cached (LRU, max entries configurable) to avoid redundant API calls.

        Raises:
            RuntimeError: If embedding_url is not configured
        """
        if not self.embedding_url:
            raise RuntimeError(
                "KB_EMBEDDING_URL not configured. Set this environment variable to your embedding endpoint."
            )

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = self._cache_get(text_hash)
        if cached is not None:
            return serialize_f32(cached)

        embedding = self._embed_remote(text)
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
