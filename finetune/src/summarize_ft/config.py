"""설정 스키마 + YAML 로드 + `--set key=value` CLI 오버라이드 + 정합성 검증.

design doc 6.3절: target_modules는 생략 가능(= modeling.py가 자동 감지)하므로
LoraConfig.target_modules 기본값은 None이다. 새 베이스 모델을 추가할 때
config에서 target_modules 줄 자체를 지워도 되도록 하기 위함.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """설정이 필수 필드를 누락했거나 타입이 맞지 않을 때 발생."""


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None  # None이면 modeling.py가 자동 감지 (6.3절)


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    epochs: int = 3
    effective_batch_size: int = 16
    max_seq_len: int = 1024
    lr_scheduler: str = "cosine"
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3  # Drive 용량 보호 — 최근 N개 체크포인트만 유지
    seed: int = 42


@dataclass
class DataConfig:
    train_path: str = ""
    val_path: str = ""
    data_version: str = "v1"


@dataclass
class Config:
    base_model: str = ""
    task: str = "summarize"  # "summarize" | "insight"
    quantization: str = "qlora_4bit"  # "qlora_4bit" | "none"
    output_dir: str = "runs/default"
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NESTED_TYPES = {"lora": LoraConfig, "train": TrainConfig, "data": DataConfig}


def _dict_to_config(d: dict[str, Any]) -> Config:
    d = dict(d)  # shallow copy, 원본 보존
    kwargs: dict[str, Any] = {}
    for key, nested_cls in _NESTED_TYPES.items():
        nested_dict = d.pop(key, {}) or {}
        if not isinstance(nested_dict, dict):
            raise ConfigError(f"'{key}' 섹션은 dict여야 함, got {type(nested_dict)}")
        valid_keys = {f.name for f in fields(nested_cls)}
        unknown = set(nested_dict) - valid_keys
        if unknown:
            raise ConfigError(f"'{key}' 섹션에 알 수 없는 필드: {sorted(unknown)}")
        if "learning_rate" in nested_dict:
            # PyYAML은 소수점 없는 지수 표기(예: 1e-4)를 float가 아니라 문자열로 파싱한다.
            # 모든 finetune/configs/*.yaml이 이 표기를 쓰므로 명시적으로 캐스팅한다.
            nested_dict["learning_rate"] = float(nested_dict["learning_rate"])
        kwargs[key] = nested_cls(**nested_dict)

    valid_top_keys = {f.name for f in fields(Config)}
    unknown_top = set(d) - valid_top_keys
    if unknown_top:
        raise ConfigError(f"알 수 없는 최상위 필드: {sorted(unknown_top)}")
    kwargs.update(d)
    return Config(**kwargs)


def load_config(path: str | Path) -> Config:
    """YAML 파일을 읽어 Config로 변환하고 검증까지 수행한다."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dict_to_config(raw)
    validate_config(cfg)
    return cfg


def _cast_value(raw: str) -> Any:
    """`--set` 값 문자열을 적당한 파이썬 타입으로 캐스팅한다."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        items = [s.strip() for s in raw[1:-1].split(",") if s.strip()]
        return items
    return raw


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """`--set train.learning_rate=1e-4 lora.r=32` 형태의 오버라이드를 적용한다.

    train.py CLI에서 사용. 원본 cfg는 변경하지 않고 새 Config를 반환한다.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set 값은 key=value 형식이어야 함: {item!r}")
        key_path, raw_value = item.split("=", 1)
        parts = key_path.strip().split(".")
        value = _cast_value(raw_value.strip())

        if len(parts) == 1:
            top = parts[0]
            if not hasattr(cfg, top):
                raise ConfigError(f"알 수 없는 설정 키: {key_path}")
            setattr(cfg, top, value)
        elif len(parts) == 2:
            section, key = parts
            target = getattr(cfg, section, None)
            if target is None or not is_dataclass(target):
                raise ConfigError(f"알 수 없는 설정 섹션: {section}")
            if not hasattr(target, key):
                raise ConfigError(f"알 수 없는 설정 키: {key_path}")
            setattr(target, key, value)
        else:
            raise ConfigError(f"지원하지 않는 키 깊이: {key_path}")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if not cfg.base_model:
        raise ConfigError("base_model은 필수")
    if cfg.task not in ("summarize", "insight"):
        raise ConfigError(f"task는 summarize|insight여야 함, got {cfg.task!r}")
    if cfg.quantization not in ("qlora_4bit", "none"):
        raise ConfigError(f"quantization은 qlora_4bit|none이어야 함, got {cfg.quantization!r}")
    if cfg.lora.r <= 0 or cfg.lora.alpha <= 0:
        raise ConfigError("lora.r / lora.alpha는 양수여야 함")
    if cfg.train.epochs <= 0:
        raise ConfigError("train.epochs는 양수여야 함")
    if cfg.train.learning_rate <= 0:
        raise ConfigError("train.learning_rate는 양수여야 함")
    if cfg.train.effective_batch_size <= 0:
        raise ConfigError("train.effective_batch_size는 양수여야 함")
    if cfg.train.max_seq_len <= 0:
        raise ConfigError("train.max_seq_len은 양수여야 함")
