"""make_train_data 파이프라인 진입점.

이 CLI는 `finetune/` 디렉터리에서 실행해야 한다 — `python -m pkg`는 CWD를 sys.path에
추가하는데, make_train_data 패키지가 finetune/ 아래 있기 때문이다.

    cd finetune
    python -m make_train_data.cli run --out-dir make_train_data/output --since 2026-08-01
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .cluster_export import export_clusters
from .clustering import cluster_articles
from .config import config
from .db import fetch_articles
from .embed import embed_texts
from .entity_extract import extract as entity_extract
from .onefact import select_onefact_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(out_dir: Path, since: str | None = None) -> int:
    articles = fetch_articles(since=since)
    if len(articles) < config.MTD_MIN_ARTICLES:
        logger.info(
            "현재 raw_articles가 %d건뿐입니다(최소 %d건 필요). data_pipeline을 더 돌려 "
            "데이터를 쌓은 뒤 다시 실행하세요.",
            len(articles), config.MTD_MIN_ARTICLES,
        )
        return 1

    clusters, unclustered = cluster_articles(
        articles,
        narrow_window_hours=config.MTD_NARROW_WINDOW_HOURS,
        broad_window_days=config.MTD_BROAD_WINDOW_DAYS,
        entity_jaccard_threshold=config.MTD_ENTITY_JACCARD_THRESHOLD,
        embed_sim_threshold=config.MTD_EMBED_SIM_THRESHOLD,
        dedup_threshold=config.MTD_DEDUP_THRESHOLD,
        min_cluster_size=config.MTD_MIN_CLUSTER_SIZE,
        entity_fn=entity_extract,
        embed_fn=embed_texts,
    )
    onefact_articles = select_onefact_candidates(
        clusters, unclustered, target_ratio=config.MTD_ONEFACT_RATIO, total_pool_size=len(articles)
    )
    paths = export_clusters(clusters, onefact_articles, out_dir=out_dir)
    logger.info("클러스터 %d개 + 단발성 배치 파일 %d개 -> %s", len(clusters), len(paths) - len(clusters), out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="군집 기반 인사이트 학습 데이터 생성")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="DB에서 기사를 읽어 군집 파일을 생성한다")
    run_parser.add_argument("--out-dir", type=Path, default=Path("make_train_data/output"))
    run_parser.add_argument("--since", default=None, help="이 날짜(YYYY-MM-DD) 이후 기사만 대상")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run(out_dir=args.out_dir, since=args.since)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
