"""학습 진입점.

사용법 (design doc 6.3절):
    python -m summarize_ft.train --config finetune/configs/qlora_qwen3-8b.yaml
    python -m summarize_ft.train --config finetune/configs/smoke.yaml --set train.epochs=1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import Config, apply_overrides, load_config
from .data import DynamicPaddingCollator, build_dataset
from .modeling import build_model_and_tokenizer
from .registry import log_run


def config_hash(cfg: Config) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_trainer(cfg: Config):
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_dataset = build_dataset(cfg.data.train_path, tokenizer, max_seq_len=cfg.train.max_seq_len)
    eval_dataset = (
        build_dataset(cfg.data.val_path, tokenizer, max_seq_len=cfg.train.max_seq_len)
        if cfg.data.val_path
        else None
    )

    # effective_batch_size = per_device_batch * grad_accum. 코랩 T4(15GB)에서
    # per_device_batch=1이 기본 전제이므로 grad_accum으로 유효 배치를 맞춘다
    # (design doc 1.1절/2절).
    per_device_batch = 1
    grad_accum = max(1, cfg.train.effective_batch_size // per_device_batch)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,  # Drive 용량 보호 — 최근 N개만 남기고 이전 체크포인트는 자동 삭제
        gradient_checkpointing=True,  # 필수 — 안 켜면 T4에서 OOM (design doc 1.1절)
        optim="paged_adamw_8bit",  # AdamW 옵티마이저 상태를 8bit로 저장 — VRAM과 체크포인트 용량을 모두 줄임(QLoRA 표준 관행)
        seed=cfg.train.seed,
        report_to=[],
        logging_steps=10,
    )

    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    return trainer, tokenizer


def run_train(cfg: Config) -> str:
    """학습을 실행하고 output_dir 경로를 반환한다. registry.py에 실행 기록을 남긴다."""
    trainer, _tokenizer = build_trainer(cfg)
    trainer.train(resume_from_checkpoint=_find_last_checkpoint(cfg.output_dir))
    trainer.save_model(cfg.output_dir)

    log_run(
        config=cfg,
        config_hash=config_hash(cfg),
        stage="train",
        metrics={},
    )
    return cfg.output_dir


def _find_last_checkpoint(output_dir: str) -> str | None:
    """구글 드라이브 등에 저장된 마지막 체크포인트가 있으면 그 경로를 반환한다.

    코랩 세션이 끊겨도 다음 세션에서 이어 학습할 수 있도록
    (design doc 1.1절: save_strategy="steps" + resume_from_checkpoint 권장).
    """
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize_ft 학습 진입점")
    parser.add_argument("--config", required=True, help="YAML config 경로")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="overrides",
        help="key.subkey=value 형태 오버라이드 (예: train.epochs=1 lora.r=32)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    output_dir = run_train(cfg)
    print(f"학습 완료. 결과: {output_dir}")


if __name__ == "__main__":
    main()
