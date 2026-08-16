"""베이스 모델 로드(4bit QLoRA 또는 bf16) + target_modules 자동 감지 + LoRA 부착.

design doc 6.3절: config.lora.target_modules가 None이면 베이스 모델의 nn.Linear
레이어 이름을 스캔해 자동으로 채운다. 새 베이스 모델(Qwen3/Gemma4/EXAONE 등)을
추가할 때 config에 target_modules를 수동으로 적지 않아도 되게 하기 위함.
"""
from __future__ import annotations

from typing import Any

from .config import Config

# lm_head, embed_tokens처럼 LoRA를 붙이면 안 되거나 붙일 필요 없는 레이어는 제외한다.
_EXCLUDE_SUBSTRINGS = ("lm_head", "embed_tokens", "embed_out")


def detect_target_modules(model: Any) -> list[str]:
    """모델의 nn.Linear 레이어들에서 마지막 이름 조각(예: q_proj)만 추출해 중복 제거한다."""
    import torch.nn as nn

    names: set[str] = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split(".")[-1]
        if any(sub in full_name for sub in _EXCLUDE_SUBSTRINGS):
            continue
        names.add(leaf)
    if not names:
        raise RuntimeError("nn.Linear 레이어를 찾지 못했습니다 — 모델 구조를 확인하세요.")
    return sorted(names)


def load_base_model(cfg: Config):
    """cfg.quantization에 따라 4bit QLoRA 또는 bf16 모델을 로드한다."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"device_map": "auto"}

    if cfg.quantization == "qlora_4bit":
        from transformers import BitsAndBytesConfig

        # 코랩 무료 T4는 bf16 미지원이라 compute dtype은 fp16으로 폴백한다
        # (design doc 1.1절 / 2절 표). A100 등 bf16 지원 GPU에서는 자동으로
        # torch.cuda.is_bf16_supported()가 True라 bf16을 쓴다.
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    return model


def attach_lora(model: Any, cfg: Config):
    """LoraConfig를 만들어 model에 peft LoRA adapter를 붙인다.

    target_modules가 명시돼 있지 않으면 detect_target_modules로 자동 감지한다.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training

    if cfg.quantization == "qlora_4bit":
        model = prepare_model_for_kbit_training(model)

    target_modules = cfg.lora.target_modules or detect_target_modules(model)

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model_and_tokenizer(cfg: Config):
    """train.py/infer.py에서 쓰는 진입점. (model, tokenizer) 튜플을 반환한다."""
    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model = attach_lora(model, cfg)
    return model, tokenizer
