import pytest

from summarize_ft.config import Config, ConfigError, apply_overrides, load_config, validate_config


def _write_yaml(tmp_path, content: str):
    path = tmp_path / "cfg.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_minimal_config(tmp_path):
    path = _write_yaml(tmp_path, "base_model: Qwen/Qwen3-8B-Instruct\n")
    cfg = load_config(path)
    assert cfg.base_model == "Qwen/Qwen3-8B-Instruct"
    assert cfg.lora.r == 16  # 기본값
    assert cfg.lora.target_modules is None  # modeling.py가 자동 감지하도록 기본 None


def test_load_full_config(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
base_model: Qwen/Qwen3-8B-Instruct
task: summarize
quantization: qlora_4bit
lora:
  r: 32
  alpha: 64
train:
  learning_rate: 1e-4
  epochs: 2
data:
  train_path: data/train.jsonl
output_dir: runs/test
""",
    )
    cfg = load_config(path)
    assert cfg.lora.r == 32
    assert cfg.train.learning_rate == 1e-4
    assert cfg.data.train_path == "data/train.jsonl"
    assert cfg.output_dir == "runs/test"


def test_missing_base_model_raises(tmp_path):
    path = _write_yaml(tmp_path, "task: summarize\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_top_level_field_raises(tmp_path):
    path = _write_yaml(tmp_path, "base_model: x\nfoo: bar\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_nested_field_raises(tmp_path):
    path = _write_yaml(tmp_path, "base_model: x\nlora:\n  unknown_key: 1\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_apply_overrides_top_level():
    cfg = Config(base_model="x")
    new_cfg = apply_overrides(cfg, ["output_dir=runs/new"])
    assert new_cfg.output_dir == "runs/new"
    assert cfg.output_dir != "runs/new"  # 원본은 안 바뀜


def test_apply_overrides_nested():
    cfg = Config(base_model="x")
    new_cfg = apply_overrides(cfg, ["train.epochs=1", "lora.r=4"])
    assert new_cfg.train.epochs == 1
    assert new_cfg.lora.r == 4


def test_apply_overrides_unknown_section_raises():
    cfg = Config(base_model="x")
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["nope.foo=1"])


def test_apply_overrides_bad_format_raises():
    cfg = Config(base_model="x")
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["not_a_kv_pair"])


@pytest.mark.parametrize("bad_field", ["r", "alpha"])
def test_invalid_lora_values_raise(bad_field):
    cfg = Config(base_model="x")
    setattr(cfg.lora, bad_field, 0)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_invalid_task_raises():
    cfg = Config(base_model="x", task="translate")
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_invalid_quantization_raises():
    cfg = Config(base_model="x", quantization="int8")
    with pytest.raises(ConfigError):
        validate_config(cfg)
