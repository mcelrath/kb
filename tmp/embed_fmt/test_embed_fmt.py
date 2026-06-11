"""Tests for EmbeddingService format abstraction (llamacpp vs openai)."""

import json
import math
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError

# Ensure kb package is importable from the project root
sys.path.insert(0, "/home/mcelrath/Projects/ai/kb")

from kb.core.embedding import EmbeddingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2norm(v):
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v
    return [x / mag for x in v]


def _approx_equal(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _parse_f32(blob):
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Mock HTTP server
# ---------------------------------------------------------------------------

# Shared state for the mock server
_server_state: dict = {}


class MockEmbeddingHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress output

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        # Record request details for assertions
        _server_state["last_path"] = self.path
        _server_state["last_body"] = body
        _server_state["last_auth"] = self.headers.get("Authorization", "")

        if self.path == "/embedding":
            # llamacpp format: {"content": text_or_list}
            content = body.get("content", "")
            if isinstance(content, list):
                # batch
                response = []
                for item in content:
                    response.append({
                        "index": len(response),
                        "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    })
            else:
                # single — return two token embeddings to exercise mean-pool
                response = [{"index": 0, "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}]
            self._send_json(200, response)

        elif self.path == "/v1/embeddings":
            # openai format
            # dimensions-reject simulation
            if _server_state.get("reject_dimensions") and "dimensions" in body:
                self._send_json(400, {"error": {"message": "unsupported dimensions field"}})
                return

            inp = body.get("input", "")
            if isinstance(inp, list):
                # batch
                data = []
                for i, _ in enumerate(inp):
                    data.append({"index": i, "embedding": [0.5, 0.5, 0.0]})
                # Return out-of-order to test sort-by-index
                self._send_json(200, {"data": list(reversed(data))})
            else:
                self._send_json(200, {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _start_mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockEmbeddingHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingFormats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server, cls.port = _start_mock_server()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    def _svc_llamacpp(self, **kw):
        return EmbeddingService(
            embedding_url=f"{self.base_url}/embedding",
            embedding_dim=3,
            embedding_format="llamacpp",
            **kw,
        )

    def _svc_openai(self, key="", **kw):
        return EmbeddingService(
            embedding_url=f"{self.base_url}/v1/embeddings",
            embedding_dim=3,
            embedding_format="openai",
            embedding_model="test-model",
            embedding_key=key,
            **kw,
        )

    # --- llamacpp path ---

    def test_llamacpp_mean_pool(self):
        """llamacpp: two token embeddings are mean-pooled before L2-norm."""
        svc = self._svc_llamacpp()
        blob = svc.embed("hello", max_retries=0)
        vec = _parse_f32(blob)
        # raw mean of [[1,0,0],[0,1,0]] = [0.5, 0.5, 0.0]
        expected = _l2norm([0.5, 0.5, 0.0])
        self.assertTrue(
            _approx_equal(vec, expected, tol=1e-5),
            f"mean-pool mismatch: got {vec}, expected {expected}",
        )

    def test_llamacpp_default_no_env(self):
        """Back-compat: with KB_EMBEDDING_FORMAT UNSET, the default is llamacpp.

        Hermetic: pops the env var (a contaminated ambient env — e.g. a stray
        `kb configure` — must not flip this contract) and reloads the import-time
        constant before asserting.
        """
        import os
        import importlib
        saved = os.environ.pop("KB_EMBEDDING_FORMAT", None)
        try:
            from kb import constants
            from kb.core import embedding as emb
            importlib.reload(constants)
            importlib.reload(emb)
            svc = emb.EmbeddingService(
                embedding_url=f"{self.base_url}/embedding",
                embedding_dim=3,
            )
            self.assertEqual(svc.embedding_format, "llamacpp")
        finally:
            if saved is not None:
                os.environ["KB_EMBEDDING_FORMAT"] = saved
            from kb import constants as _c
            from kb.core import embedding as _e
            importlib.reload(_c)
            importlib.reload(_e)

    # --- openai path ---

    def test_openai_no_pooling(self):
        """openai: returned vector is used directly, NOT mean-pooled."""
        svc = self._svc_openai()
        blob = svc.embed("world", max_retries=0)
        vec = _parse_f32(blob)
        # mock returns [1.0, 0.0, 0.0] which after L2-norm is still [1, 0, 0]
        expected = _l2norm([1.0, 0.0, 0.0])
        self.assertTrue(
            _approx_equal(vec, expected, tol=1e-5),
            f"openai no-pool mismatch: got {vec}, expected {expected}",
        )

    def test_openai_bearer_auth(self):
        """openai: Authorization header is sent when embedding_key is set."""
        svc = self._svc_openai(key="sk-test-secret")
        svc.embed("auth test", max_retries=0)
        self.assertEqual(
            _server_state.get("last_auth"),
            "Bearer sk-test-secret",
            f"Expected Bearer header, got: {_server_state.get('last_auth')!r}",
        )

    def test_openai_no_auth_when_no_key(self):
        """openai: no Authorization header when key is empty."""
        svc = self._svc_openai(key="")
        svc.embed("no auth", max_retries=0)
        self.assertEqual(_server_state.get("last_auth"), "")

    # --- batch ---

    def test_openai_batch(self):
        """openai batch: returns both embeddings in input order (sorted by index)."""
        svc = self._svc_openai()
        texts = ["first", "second"]
        results = svc.embed_batch(texts)
        self.assertEqual(len(results), 2)
        # both embeddings from mock are [0.5, 0.5, 0.0] -> normalized
        expected = _l2norm([0.5, 0.5, 0.0])
        for i, blob in enumerate(results):
            vec = _parse_f32(blob)
            self.assertTrue(
                _approx_equal(vec, expected, tol=1e-5),
                f"batch item {i} mismatch: {vec} vs {expected}",
            )

    def test_llamacpp_batch(self):
        """llamacpp batch: both embeddings mean-pooled + normalized."""
        svc = self._svc_llamacpp()
        texts = ["a", "b"]
        results = svc.embed_batch(texts)
        self.assertEqual(len(results), 2)
        # mock returns two token embeddings [[0.1,0.2,0.3],[0.4,0.5,0.6]] -> mean [0.25,0.35,0.45]
        expected = _l2norm([0.25, 0.35, 0.45])
        for i, blob in enumerate(results):
            vec = _parse_f32(blob)
            self.assertTrue(
                _approx_equal(vec, expected, tol=1e-5),
                f"llamacpp batch item {i} mismatch: {vec} vs {expected}",
            )

    # --- dimensions-reject retry ---

    def test_openai_dimensions_reject_retry(self):
        """openai: 400 mentioning 'dimensions' triggers retry without that field."""
        _server_state["reject_dimensions"] = True
        try:
            svc = self._svc_openai()
            # The first call sends dimensions; mock returns 400; retry without -> 200
            blob = svc.embed("dim retry", max_retries=0)
            vec = _parse_f32(blob)
            expected = _l2norm([1.0, 0.0, 0.0])
            self.assertTrue(
                _approx_equal(vec, expected, tol=1e-5),
                f"dimensions-retry mismatch: {vec}",
            )
            # Verify the retry body did NOT contain dimensions
            self.assertNotIn("dimensions", _server_state.get("last_body", {}),
                             "Retry should omit 'dimensions'")
        finally:
            _server_state["reject_dimensions"] = False


if __name__ == "__main__":
    unittest.main(verbosity=2)
