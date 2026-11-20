import pytest

import cartblanche.agent.retrieval as retrieval


@pytest.fixture(autouse=True)
def _embedding_cache_in_tmp(tmp_path, monkeypatch):
    """Tests never write embedding vectors into the repo's cache/ folder."""
    monkeypatch.setattr(retrieval, "CACHE_DIR", tmp_path / "embeddings")
