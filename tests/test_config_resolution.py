"""
Characterization tests for embedding configuration resolution.

Asserts how KB_EMBEDDING_URL / FORMAT / MODEL / DIM env vars flow into
EmbeddingService via constants.py. This pins the behavior R3 must preserve.

Key invariants from constants.py:
  DEFAULT_EMBEDDING_URL    = env.get("KB_EMBEDDING_URL", "http://ash:8081/embedding")
  DEFAULT_EMBEDDING_DIM    = int(env.get("KB_EMBEDDING_DIM", "4096"))
  DEFAULT_EMBEDDING_FORMAT = env.get("KB_EMBEDDING_FORMAT", "llamacpp")
  DEFAULT_EMBEDDING_MODEL  = env.get("KB_EMBEDDING_MODEL", "")
  DEFAULT_EMBEDDING_KEY    = env.get("KB_EMBEDDING_KEY", "")

And EmbeddingService.__init__ accepts these as kwargs with those defaults.
"""

import importlib
import os
import sys
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_constants_with_env(env_overrides: dict) -> object:
    """Import kb.constants with a patched os.environ, return the module."""
    # We patch os.environ.get at the module level by reloading with overrides.
    patched_env = {**os.environ, **env_overrides}
    # Remove keys not in overrides that should be absent
    for k in list(patched_env.keys()):
        if k.startswith("KB_") and k not in env_overrides:
            del patched_env[k]

    with patch.dict(os.environ, env_overrides, clear=False):
        # Force reload so module-level os.environ.get() re-executes
        if "kb.constants" in sys.modules:
            del sys.modules["kb.constants"]
        if "kb.core.embedding" in sys.modules:
            del sys.modules["kb.core.embedding"]
        import kb.constants as c
        return c


def _make_service_with_env(env_overrides: dict):
    """Construct EmbeddingService with patched env (re-imports constants)."""
    with patch.dict(os.environ, env_overrides, clear=False):
        if "kb.constants" in sys.modules:
            del sys.modules["kb.constants"]
        if "kb.core.embedding" in sys.modules:
            del sys.modules["kb.core.embedding"]
        from kb.core.embedding import EmbeddingService
        return EmbeddingService()


# ---------------------------------------------------------------------------
# constants.py defaults (no env set)
# ---------------------------------------------------------------------------

class TestDefaultsNoEnv:
    """With no KB_* env vars set, constants resolve to hardcoded defaults."""

    def _clean_env(self):
        return {k: "" for k in [
            "KB_EMBEDDING_URL", "KB_EMBEDDING_DIM", "KB_EMBEDDING_FORMAT",
            "KB_EMBEDDING_MODEL", "KB_EMBEDDING_KEY",
        ]}

    def test_default_url(self):
        env = self._clean_env()
        env["KB_EMBEDDING_URL"] = ""
        with patch.dict(os.environ, {}, clear=False):
            for k in ["KB_EMBEDDING_URL", "KB_EMBEDDING_DIM",
                      "KB_EMBEDDING_FORMAT", "KB_EMBEDDING_MODEL", "KB_EMBEDDING_KEY"]:
                os.environ.pop(k, None)
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_URL == "http://ash:8081/embedding"

    def test_default_dim(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_DIM", None)
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_DIM == 4096

    def test_default_format(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_FORMAT", None)
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_FORMAT == "llamacpp"

    def test_default_model_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_MODEL", None)
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_MODEL == ""

    def test_default_key_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_KEY", None)
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_KEY == ""


# ---------------------------------------------------------------------------
# constants.py with env overrides
# ---------------------------------------------------------------------------

class TestEnvOverridesConstants:
    """Env vars flow into constants at import time."""

    def test_url_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_URL": "http://myhost:1234/embed"}):
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_URL == "http://myhost:1234/embed"

    def test_dim_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_DIM": "1024"}):
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_DIM == 1024

    def test_format_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_FORMAT": "openai"}):
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_FORMAT == "openai"

    def test_model_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_MODEL": "qwen3-embedding"}):
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_MODEL == "qwen3-embedding"

    def test_key_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_KEY": "sk-test-key"}):
            if "kb.constants" in sys.modules:
                del sys.modules["kb.constants"]
            import kb.constants as c
            assert c.DEFAULT_EMBEDDING_KEY == "sk-test-key"


# ---------------------------------------------------------------------------
# EmbeddingService picks up constants as defaults
# ---------------------------------------------------------------------------

class TestEmbeddingServiceDefaultsFromConstants:
    """EmbeddingService() with no kwargs reads from constants (which read from env)."""

    def _clean_modules(self):
        for mod in ["kb.constants", "kb.core.embedding"]:
            sys.modules.pop(mod, None)

    def test_service_url_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_URL": "http://testhost:9999/emb"}):
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_url == "http://testhost:9999/emb"

    def test_service_dim_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_DIM": "768"}):
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_dim == 768

    def test_service_format_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_FORMAT": "openai"}):
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_format == "openai"

    def test_service_model_from_env(self):
        with patch.dict(os.environ, {"KB_EMBEDDING_MODEL": "nomic-embed-text"}):
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_model == "nomic-embed-text"

    def test_service_url_default_ash(self):
        """Without env, url defaults to ash:8081."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_URL", None)
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_url == "http://ash:8081/embedding"

    def test_service_dim_default_4096(self):
        """Without env, dim defaults to 4096."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_DIM", None)
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_dim == 4096

    def test_service_format_default_llamacpp(self):
        """Without env, format defaults to llamacpp."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KB_EMBEDDING_FORMAT", None)
            self._clean_modules()
            from kb.core.embedding import EmbeddingService
            svc = EmbeddingService()
            assert svc.embedding_format == "llamacpp"


# ---------------------------------------------------------------------------
# EmbeddingService kwargs override constants
# ---------------------------------------------------------------------------

class TestEmbeddingServiceKwargOverride:
    """Explicit kwargs to EmbeddingService() override the constant defaults."""

    def test_explicit_url_override(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService(embedding_url="http://override:1111/emb")
        assert svc.embedding_url == "http://override:1111/emb"

    def test_explicit_dim_override(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService(embedding_dim=512)
        assert svc.embedding_dim == 512

    def test_explicit_format_override(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService(embedding_format="openai")
        assert svc.embedding_format == "openai"

    def test_explicit_model_override(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService(embedding_model="my-model")
        assert svc.embedding_model == "my-model"

    def test_explicit_key_override(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService(embedding_key="sk-xyz")
        assert svc.embedding_key == "sk-xyz"

    def test_cache_max_default(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService()
        assert svc._cache_max == 500

    def test_cache_starts_empty(self):
        from kb.core.embedding import EmbeddingService
        svc = EmbeddingService()
        assert svc._cache == {}
        assert svc._cache_order == []
