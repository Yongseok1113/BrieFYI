"""sources/* 를 호출해 정제→분할까지 수행하는 오케스트레이션 스크립트 (design doc 6.1/6.7절).

6.7절에서 열어뒀던 질문("prepare_data.py가 다중 소스를 지원하는가")에 대한 답: 이 스크립트는
처음부터 다중 소스 전제로 짰다. --sources로 켤 소스를 고르고, 소스별로 sources/*.py의
로더를 호출한 뒤 하나의 공통 스키마 리스트로 합치고, 길이 필터링 후 train/val로 나눈다.

    python finetune/scripts/prepare_data.py --sources digest_pipeline aihub dacon \
        --aihub-dir /path/to/aihub --dacon-csv /path/to/dacon/train.csv \
        --out-dir finetune/data/processed
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "finetune" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from summarize_ft.jsonl import write_jsonl  # noqa: E402
from summarize_ft.schema import SchemaError, validate_example  # noqa: E402

MIN_ARTICLE_CHARS = 30
MAX_ARTICLE_CHARS = 8000  # 3.3절: 원문 길이 편차가 큰 기사는 토큰 제한에 맞게 필터링


def _load_digest_pipeline(min_date: str | None) -> list[dict]:
    from summarize_ft.sources.digests_export import export_examples

    return export_examples(min_digest_date=min_date)


def _load_aihub(aihub_dir: str) -> list[dict]:
    from summarize_ft.sources.aihub_loader import load_dir

    return load_dir(aihub_dir)


def _load_dacon(dacon_csv: str) -> list[dict]:
    from summarize_ft.sources.dacon_loader import load_csv

    return load_csv(dacon_csv)


SOURCE_LOADERS = {
    "digest_pipeline": _load_digest_pipeline,
    "aihub": _load_aihub,
    "dacon": _load_dacon,
}


def clean(examples: list[dict]) -> list[dict]:
    """3.3절 데이터 정제: 길이 필터링 + 스키마 검증 통과분만 채택."""
    cleaned = []
    for ex in examples:
        text = ex.get("input", {}).get("article_text", "")
        if ex["task"] == "summarize" and not (MIN_ARTICLE_CHARS <= len(text) <= MAX_ARTICLE_CHARS):
            continue
        try:
            validate_example(ex)
        except SchemaError:
            continue
        cleaned.append(ex)
    return cleaned


def split_train_val(examples: list[dict], val_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="다중 소스 학습 데이터 준비")
    parser.add_argument(
        "--sources", nargs="+", default=["digest_pipeline"],
        choices=list(SOURCE_LOADERS), help="사용할 데이터 소스 (복수 지정 가능)"
    )
    parser.add_argument("--min-date", default=None, help="digest_pipeline 소스에서 이 날짜 이후만 export")
    parser.add_argument("--aihub-dir", default=None)
    parser.add_argument("--dacon-csv", default=None)
    parser.add_argument("--out-dir", default="finetune/data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    all_examples: list[dict] = []
    for source in args.sources:
        if source == "digest_pipeline":
            all_examples.extend(_load_digest_pipeline(args.min_date))
        elif source == "aihub":
            if not args.aihub_dir:
                raise SystemExit("--sources aihub를 쓰려면 --aihub-dir이 필요합니다")
            all_examples.extend(_load_aihub(args.aihub_dir))
        elif source == "dacon":
            if not args.dacon_csv:
                raise SystemExit("--sources dacon을 쓰려면 --dacon-csv가 필요합니다")
            all_examples.extend(_load_dacon(args.dacon_csv))

    print(f"소스별 로드 후 총 {len(all_examples)}건")

    cleaned = clean(all_examples)
    print(f"정제 후 {len(cleaned)}건 (제외 {len(all_examples) - len(cleaned)}건)")

    train, val = split_train_val(cleaned, val_ratio=args.val_ratio)

    out_dir = Path(args.out_dir)
    train_path = out_dir / "summarize_train.jsonl"
    val_path = out_dir / "summarize_val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    print(f"train {len(train)}건 -> {train_path}")
    print(f"val   {len(val)}건 -> {val_path}")


if __name__ == "__main__":
    main()
