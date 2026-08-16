# finetune/ 아키텍처

원 설계 문서: `docs/lora-finetune-summarization-design.md` (6절이 이 코드의 청사진).

## 데이터 흐름

```
sources/*.py (원본 포맷별 파서)
    -> 공통 JSONL 스키마 (schema.py)
        -> scripts/prepare_data.py (정제 + train/val 분할)
            -> data/processed/*.jsonl
                -> data.py (completion-only 마스킹 + 동적 패딩)
                    -> train.py (SFTTrainer)
                        -> runs/<name>/ (LoRA adapter 체크포인트)
                            -> evaluate.py (4단계 평가)
                            -> merge_lora.py (선택: 병합 배포)
```

## 모듈 책임

| 모듈 | 책임 | 무거운 의존성 |
| --- | --- | --- |
| `schema.py` | 공통 예제 스키마 정의 + 검증 | 없음 |
| `jsonl.py` | JSONL 입출력 | 없음 |
| `config.py` | YAML 로드, `--set` 오버라이드, 정합성 검증 | pyyaml만 |
| `prompt.py` | chat message 구성, 토큰 절단 | 없음 (tokenizer는 선택 인자) |
| `data.py` | completion-only 마스킹, 동적 패딩 콜레이터 | torch (지연 import) |
| `modeling.py` | 4bit 로드, target_modules 자동 감지, LoRA 부착 | transformers/peft/bitsandbytes |
| `train.py` | 학습 진입점 (CLI) | 위 전부 + trl |
| `infer.py` | adapter로 추론 (agents/summarizer.py의 summarize_hf 자리에 꽂힘) | transformers/peft |
| `evaluate.py` | 4단계 평가 | rouge-score/bert-score (선택), anthropic (선택) |
| `merge_lora.py` | adapter -> 단일 체크포인트 병합 | transformers/peft |
| `registry.py` | 실행 기록 (JSONL, 필요시 Postgres) | 없음 |
| `sources/*.py` | 원본 -> 공통 스키마 변환 | 소스별 상이 (digests_export만 psycopg 필요) |

의도적으로 `schema.py`/`jsonl.py`/`config.py`/`prompt.py`/`registry.py`는 torch 없이 import 가능하게
만들었다 — GPU 없는 CI나 로컬에서도 `pytest finetune/tests`가 항상 돌아가게 하기 위함.

## BrieFYI 본체와의 접점

- `sources/digests_export.py`가 repo 루트의 `db/db.py`, `config.py`를 그대로 재사용한다(스키마 중복 정의 방지).
- `infer.py`의 `summarize_hf`/`insight_hf`는 `tools/summarize.py`/`tools/insight.py`와 동일한
  입출력 시그니처를 맞춰뒀다. 검증이 끝나면 `agents/registry.py`에 `summarize_hf`를 등록하고
  `SUMMARIZER_PROVIDER=hf`로 전환하면 된다 (`agent-management-structure.md`, design doc 5절 참고).
- `schema.py`의 output 스키마는 `tools/summarize.py`/`tools/insight.py`가 실제로 뱉는 JSON 형식과
  1:1로 맞춰져 있다 — 학습 데이터와 프로덕션 출력이 항상 같은 형식이 되도록.

## target_modules 자동 감지 (6.3절)

`modeling.py:detect_target_modules`가 베이스 모델의 `nn.Linear` 레이어 이름을 스캔해
`target_modules`를 자동으로 채운다. `configs/*.yaml`에서 `lora.target_modules`를 생략하면
이 로직이 켜진다. 새 베이스 모델을 추가할 때 config에 손댈 부분이 하나 줄어든다.
