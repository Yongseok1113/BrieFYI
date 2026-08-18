"""rag_latest.embed.embed_texts()를 그대로 호출하는 얇은 래퍼."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_latest.embed import embed_texts as _embed_texts  # noqa: E402


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _embed_texts(texts)
