"""CLI 진입점.

    python -m data_pipeline.cli run --stage all --limit 20
    python -m data_pipeline.cli run --stage extract --limit 50
    python -m data_pipeline.cli synonyms build
"""
from __future__ import annotations

import argparse
import json


def _cmd_run(args: argparse.Namespace) -> None:
    from . import extract, ingest, normalize, enrich as enrich_stage, pipeline
    from .sources.gnews_source import GNewsSource
    from .sources.naver_source import NaverNewsSource

    source = NaverNewsSource() if args.source == "naver" else GNewsSource()
    fetch_kwargs = {"max_results": args.max_results} if args.max_results else {}

    if args.stage == "all":
        # run_all은 아직 source 선택을 안 받는다 -- ingest만 네이버로 돌리려면 --stage ingest 사용.
        result = pipeline.run_all(limit=args.limit, do_ingest=not args.no_ingest, keywords=args.keywords)
    elif args.stage == "ingest":
        result = (
            pipeline.run_ingest_multi(args.keywords, source=source, **fetch_kwargs)
            if args.keywords
            else ingest.run_ingest(source, **fetch_kwargs)
        )
    elif args.stage == "extract":
        result = extract.run_extract(args.limit)
    elif args.stage == "enrich":
        result = enrich_stage.run_enrich(args.limit)
    elif args.stage == "normalize":
        result = normalize.run_normalize(args.limit)
    else:
        raise SystemExit(f"알 수 없는 stage: {args.stage}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_synonyms(args: argparse.Namespace) -> None:
    from . import synonym_builder

    if args.synonyms_action == "build":
        result = synonym_builder.build_all()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        raise SystemExit(f"알 수 없는 synonyms 액션: {args.synonyms_action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="data_pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="파이프라인 단계 실행")
    run_parser.add_argument("--stage", default="all", choices=["all", "ingest", "extract", "enrich", "normalize"])
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--no-ingest", action="store_true", help="--stage all일 때 수집은 건너뛰기")
    run_parser.add_argument(
        "--keywords", nargs="+", default=None,
        help="ingest/all 단계에서 여러 키워드로 나눠 수집 (예: --keywords 경제 산업 금융 기술)",
    )
    run_parser.add_argument(
        "--source", default="gnews", choices=["gnews", "naver"],
        help="ingest 단계에서 쓸 소스 (기본 gnews)",
    )
    run_parser.add_argument(
        "--max-results", type=int, default=None,
        help="ingest 단계에서 키워드당 수집할 기사 수 (미지정 시 NEWS_MAX_RESULTS)",
    )
    run_parser.set_defaults(func=_cmd_run)

    synonyms_parser = subparsers.add_parser("synonyms", help="통합 단어 테이블 관리")
    synonyms_parser.add_argument("synonyms_action", choices=["build"])
    synonyms_parser.set_defaults(func=_cmd_synonyms)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
