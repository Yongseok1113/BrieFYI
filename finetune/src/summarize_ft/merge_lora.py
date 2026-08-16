"""LoRA adapter를 베이스 모델에 병합해 단일 체크포인트로 저장.

Hugging Face Inference Endpoints처럼 adapter를 따로 못 얹는 배포 방식을 쓸 때,
혹은 추론 속도를 최대화하고 싶을 때 사용한다. 반대로 vLLM/TGI 멀티 adapter
서빙을 쓸 계획이면 병합하지 않고 adapter 그대로 Hugging Face Hub에 올리는 쪽이
더 유연하다 (design doc 5절).

사용법:
    python -m summarize_ft.merge_lora --base Qwen/Qwen3-8B-Instruct \
        --adapter runs/qwen3-8b-summarize-v1 --out runs/qwen3-8b-summarize-v1-merged
"""
from __future__ import annotations

import argparse


def merge_and_save(base_model_name: str, adapter_path: str, out_path: str,
                    *, offload_folder: str | None = None) -> None:
    """LoRA adapter를 베이스에 병합해 out_path에 저장한다.

    8B급 모델을 bf16으로 로드하면 가중치만 ~16GB라 코랩 무료 T4(VRAM 15GB)나
    기본 시스템 RAM(~12GB대)에 통째로 안 들어갈 수 있다. device_map="auto" +
    low_cpu_mem_usage=True로 accelerate가 GPU/CPU/디스크에 자동으로 나눠 싣게 하고,
    그래도 부족하면 offload_folder로 일부를 디스크에 내려서 처리한다.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if offload_folder:
        load_kwargs["offload_folder"] = offload_folder
        load_kwargs["offload_state_dict"] = True

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained(out_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(out_path)
    print(f"병합 완료: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA adapter를 베이스 모델에 병합")
    parser.add_argument("--base", required=True, help="베이스 모델 이름 (예: Qwen/Qwen3-8B-Instruct)")
    parser.add_argument("--adapter", required=True, help="LoRA adapter 경로")
    parser.add_argument("--out", required=True, help="병합된 모델을 저장할 경로")
    parser.add_argument("--offload-folder", default=None, help="메모리가 부족할 때 디스크로 offload할 임시 경로")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merge_and_save(args.base, args.adapter, args.out, offload_folder=args.offload_folder)


if __name__ == "__main__":
    main()
