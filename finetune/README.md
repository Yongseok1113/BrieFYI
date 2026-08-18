# finetune/

BrieFYI 요약/인사이트 모델을 LoRA/QLoRA로 파인튜닝해 Hugging Face에 배포하기 위한 패키지.

설계 문서: `docs/lora-finetune-summarization-design.md`
사용법: `finetune/docs/USAGE.md`
아키텍처: `finetune/docs/ARCHITECTURE.md`

빠른 시작:

```bash
cd finetune
bash scripts/setup.sh
python scripts/check_env.py
python -m summarize_ft.train --config configs/smoke.yaml   # 스모크 테스트
pytest tests
```

`make_train_data`는 `finetune/` 디렉터리에서 실행해야 한다: `cd finetune && python -m make_train_data.cli run`.
