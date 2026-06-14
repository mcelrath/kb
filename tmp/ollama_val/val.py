"""
Validation: EmbeddingService openai-format path against local Ollama.

Tests:
  1. embed_raw() -> list[float], len==768, L2-normalized
  2. embed() -> bytes
  3. embed_batch(["a","b"]) -> 2 results
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from kb.core.embedding import EmbeddingService

OLLAMA_URL = "http://localhost:11434/v1/embeddings"
MODEL = "nomic-embed-text"
DIM = 768

svc = EmbeddingService(
    embedding_url=OLLAMA_URL,
    embedding_dim=DIM,
    embedding_format="openai",
    embedding_model=MODEL,
)

print(f"EmbeddingService: url={OLLAMA_URL} model={MODEL} dim={DIM} format=openai")

# Test 1: embed_raw
print("\n--- Test 1: embed_raw ---")
v = svc.embed_raw("knowledge base smoke test")
print(f"  len(v)     = {len(v)}")
norm_sq = sum(x * x for x in v)
print(f"  |v|^2      = {norm_sq:.6f}  (should be ~1.0)")
assert len(v) == DIM, f"Expected dim={DIM}, got {len(v)}"
assert abs(norm_sq - 1.0) < 1e-3, f"Not L2-normalized: |v|^2={norm_sq}"
print("  PASS: correct dim + L2-normalized")

# Test 2: embed (returns bytes)
print("\n--- Test 2: embed ---")
b = svc.embed("knowledge base smoke test")
print(f"  type(b)    = {type(b)}")
print(f"  len(bytes) = {len(b)}  (expect {DIM*4} for float32)")
assert isinstance(b, bytes), f"Expected bytes, got {type(b)}"
assert len(b) == DIM * 4, f"Expected {DIM*4} bytes, got {len(b)}"
print("  PASS: returns bytes of correct length")

# Test 3: embed_batch
print("\n--- Test 3: embed_batch ---")
results = svc.embed_batch(["hello world", "knowledge base"])
print(f"  len(results) = {len(results)}  (expect 2)")
assert len(results) == 2, f"Expected 2 results, got {len(results)}"
for i, r in enumerate(results):
    assert isinstance(r, bytes), f"Result {i} not bytes"
    assert len(r) == DIM * 4, f"Result {i} wrong byte length: {len(r)}"
print("  PASS: 2 serialized embeddings returned")

print("\n=== ALL TESTS PASSED ===")
print(f"Confirmed: EmbeddingService format=openai works against Ollama {MODEL}")
print(f"  URL:    {OLLAMA_URL}")
print(f"  Model:  {MODEL}")
print(f"  Dim:    {DIM}")
print(f"  Sample vector first 5 floats: {v[:5]}")
