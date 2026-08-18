"""RAG CLI 진입점.

    python -m rag.cli run --keyword AI --max-results 10
    python -m rag.cli index --limit 20
    python -m rag.cli index --article-ids 19 20 --device cpu
    python -m rag.cli events --article-ids 19 20 --force
    python -m rag.cli search --query "Claude" --mode hybrid --top-k 5

`index`/`events`/`run`은 GLiNER2와 transformers가 설치된 worker 환경에서 실행한다.
`search`는 embedding API와 DB만 있으면 된다.
"""
from __future__ import annotations

import argparse
import json


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _cmd_run(args: argparse.Namespace) -> int:
    from db.db import init_db

    from . import pipeline

    init_db()
    _print(
        pipeline.run_all(
            args.keyword,
            args.days,
            args.max_results,
            device=args.device,
        )
    )
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from db.db import init_db

    from . import indexer

    init_db()
    if args.article_ids is not None:
        results = indexer.index_articles(args.article_ids, device=args.device)
    else:
        results = indexer.index_all_articles(
            args.limit if args.limit is not None else 20,
            device=args.device,
        )
    _print(results)
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    from db.db import init_db

    from . import indexer

    init_db()
    results = indexer.index_events(
        args.article_ids,
        force=args.force,
        device=args.device,
    )
    _print(results)
    return 1 if any(result["status"] == "failed" for result in results) else 0


def _cmd_search(args: argparse.Namespace) -> int:
    from . import retriever

    query = args.query if args.query is not None else input("검색어를 입력하세요: ")
    _print(
        retriever.retrieve(
            query=query,
            top_k=args.top_k,
            search_mode=args.mode,
            vector_weight=args.vector_weight,
            text_weight=args.text_weight,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            category=args.category,
            domains=args.domains,
            category_boost=args.category_boost,
            domain_boost=args.domain_boost,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrieFYI RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="GNews 수집 + 4-Layer indexing")
    run_parser.add_argument("--keyword", required=True)
    run_parser.add_argument("--days", type=int, default=1)
    run_parser.add_argument("--max-results", type=int, default=10)
    run_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run_parser.set_defaults(func=_cmd_run)

    index_parser = subparsers.add_parser("index", help="기사 본문 chunk/embedding/4-Layer indexing")
    index_group = index_parser.add_mutually_exclusive_group()
    index_group.add_argument("--article-ids", nargs="+", type=int)
    index_group.add_argument("--limit", type=int, help="embedding이 없는 기사 중 처리할 건수")
    index_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    index_parser.set_defaults(func=_cmd_index)

    events_parser = subparsers.add_parser("events", help="기존 chunk의 구조화 Event indexing")
    events_parser.add_argument("--article-ids", nargs="+", type=int, required=True)
    events_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    events_parser.add_argument("--force", action="store_true")
    events_parser.set_defaults(func=_cmd_events)

    search_parser = subparsers.add_parser("search", help="vector/text/hybrid 검색")
    search_parser.add_argument("--query", help="검색 문자열. 생략하면 터미널에서 입력")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--mode", choices=("vector", "text", "hybrid"), default="vector")
    search_parser.add_argument("--vector-weight", type=float, default=0.7)
    search_parser.add_argument("--text-weight", type=float, default=0.3)
    search_parser.add_argument("--candidate-k", type=int)
    search_parser.add_argument("--rrf-k", type=int, default=60)
    search_parser.add_argument("--category")
    search_parser.add_argument("--domain", dest="domains", action="append")
    search_parser.add_argument("--category-boost", type=float, default=0.05)
    search_parser.add_argument("--domain-boost", type=float, default=0.05)
    search_parser.set_defaults(func=_cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
