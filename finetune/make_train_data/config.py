"""레포 루트 .env를 로드하는 설정. finetune/src/summarize_ft/sources/*_export.py와
동일한 sys.path 트릭을 쓴다 — data_pipeline/과 달리 별도 이미지로 배포하지 않으므로
DB 접속 정보(config.DATABASE_URL 등)는 레포 루트 config.py에서 그대로 가져다 쓴다.
여기 정의하는 값은 make_train_data 고유 설정(MTD_ 접두사)뿐이다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")


class Config:
    MTD_NARROW_WINDOW_HOURS = float(os.getenv("MTD_NARROW_WINDOW_HOURS", "72"))
    MTD_BROAD_WINDOW_DAYS = float(os.getenv("MTD_BROAD_WINDOW_DAYS", "28"))
    MTD_ENTITY_JACCARD_THRESHOLD = float(os.getenv("MTD_ENTITY_JACCARD_THRESHOLD", "0.3"))
    MTD_EMBED_SIM_THRESHOLD = float(os.getenv("MTD_EMBED_SIM_THRESHOLD", "0.75"))
    MTD_DEDUP_THRESHOLD = float(os.getenv("MTD_DEDUP_THRESHOLD", "0.9"))
    MTD_MIN_CLUSTER_SIZE = int(os.getenv("MTD_MIN_CLUSTER_SIZE", "2"))
    MTD_ONEFACT_RATIO = float(os.getenv("MTD_ONEFACT_RATIO", "0.175"))
    MTD_MIN_ARTICLES = int(os.getenv("MTD_MIN_ARTICLES", "20"))


config = Config()
