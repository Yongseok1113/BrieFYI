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

    if args.stage == "all":
        result = pipeline.run_all(limit=args.limit, do_ingest=not args.no_ingest)
    elif args.stage == "ingest":
        result = ingest.run_ingest(GNewsSource())
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
