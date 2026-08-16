"""GPU/CUDA/필수 패키지 버전을 미리 점검한다.

"학습 몇 시간 돌리다 중간에 환경 문제로 실패"하는 걸 막기 위한 사전 체크
(design doc 6.4절). 코랩 세션 시작 직후, 또는 로컬/서버에 처음 셋업할 때 실행.

    python finetune/scripts/check_env.py
"""
from __future__ import annotations

import importlib
import sys

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "yaml",
]

OPTIONAL_PACKAGES = ["rouge_score", "bert_score", "anthropic"]


def check_packages(names: list[str]) -> dict[str, str]:
    results = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            results[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            results[name] = "MISSING"
    return results


def check_gpu() -> dict[str, str]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": "torch 미설치"}

    info = {"cuda_available": str(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = str(torch.cuda.device_count())
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info["vram_gb"] = f"{total_mem_gb:.1f}"
        info["bf16_supported"] = str(torch.cuda.is_bf16_supported())
    return info


def main() -> int:
    print("=== 필수 패키지 ===")
    required = check_packages(REQUIRED_PACKAGES)
    missing = [name for name, version in required.items() if version == "MISSING"]
    for name, version in required.items():
        print(f"  {name}: {version}")

    print("\n=== 선택 패키지 (evaluate.py 1/3단계용) ===")
    for name, version in check_packages(OPTIONAL_PACKAGES).items():
        print(f"  {name}: {version}")

    print("\n=== GPU ===")
    for key, value in check_gpu().items():
        print(f"  {key}: {value}")

    if missing:
        print(f"\n[FAIL] 필수 패키지 누락: {missing}")
        print("scripts/setup.sh (또는 setup.ps1)을 먼저 실행하세요.")
        return 1

    print("\n[OK] 필수 패키지 확인 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
