# 요약 모델 LoRA 파인튜닝 & Hugging Face 배포 설계

BrieFYI 파이프라인의 `tools/summarize.py`/`tools/insight.py`가 지금은 Claude API를 호출한다. 이 문서는 요약(및 인사이트 추출) 역할을 대체하거나 보완할 수 있는 오픈웨이트 모델을 LoRA로 파인튜닝해 Hugging Face에 배포하고 API로 쓰는 방안을 다룬다. 목적은 비용 절감과 자체 운영 통제이며, Claude를 완전히 대체하기보다는 우선 병행 운영(A/B)하며 품질을 검증하는 것을 전제로 설계한다.

## 1. 베이스 모델 추천

한국어 뉴스 요약이 핵심 태스크이므로 한국어 처리 품질, 상업적 이용 가능한 라이선스, LoRA 파인튜닝 생태계(TRL/PEFT/Unsloth 지원 여부) 세 가지를 기준으로 골랐다.

| 모델 | 크기 | 라이선스 | 한국어 품질 | 비고 |
| --- | --- | --- | --- | --- |
| **Qwen3-8B** (기본 추천) | 8B | Apache 2.0 (상업적 이용 명확) | 우수 (다국어 토크나이저가 한국어 처리에 강함) | Qwen2.5보다 상위 성능, 동일 라이선스. Unsloth/TRL/vLLM 지원이 가장 폭넓고 파인튜닝 사례·문서가 제일 많아 리스크가 낮다 |
| **Gemma 4 12B** (성능 우선 대안) | 12B (dense, unified multimodal) | Apache 2.0 (2026.4부터 순정 전환, 이전 버전의 Gemma Terms of Use 제약 없음) | Gemma 라인 특유의 넓은 다국어 커버리지 계승(3 기준 140+ 언어) | Qwen3-8B보다 파라미터가 크고 라이선스도 동급으로 자유로움. 출시된 지 얼마 안 돼 한국어 파인튜닝 사례는 Qwen만큼 쌓이지 않음 — 본 학습 전 zero-shot 한국어 요약 품질을 Qwen3-8B와 가볍게 비교 권장 |
| Mistral NeMo 12B (라이선스 백업안) | 12B | Apache 2.0 | 유럽어 위주 학습 이력이 있어 한국어는 상대적으로 약할 가능성 | 라이선스는 확실하나 한국어 품질은 실측 필요 |
| Kakao Kanana-1.5-8B | 8B | Apache 2.0 | 한국어 특화 (KoGPT 계열 자산 기반) | 한국어 뉴스체에 더 자연스러울 수 있으나 생태계/사례가 Qwen보다 적음 |
| LG EXAONE 3.5/4.0-7.8B | 7.8B | EXAONE AI Model License (자체 라이선스, 상업적 이용 조건 별도 확인 필요) | 한/영 균형에 최적화 | 성능은 좋지만 라이선스 조항(재배포·상업적 이용 제한 여부)을 반드시 먼저 확인 |

**기본 추천은 Qwen3-8B다.** 라이선스가 명확히 상업적으로 자유롭고, LoRA/QLoRA 파인튜닝 자료가 가장 풍부해 학습·운영 리스크가 낮다. 파라미터를 좀 더 키워 성능을 우선하고 싶다면 **Gemma 4 12B**가 동급 라이선스에 크기만 큰 자연스러운 대안이다(코랩 무료 T4에서 여전히 QLoRA로 돌아가지만 8B보다 빠듯함 — 2.1절 참고). Mistral NeMo 12B는 라이선스는 동일하게 깨끗하지만 한국어 강점이 상대적으로 불확실해 3순위로 둔다. 한국어 문체를 더 살리고 싶다면 별도로 Kanana-1.5-8B를 같은 데이터로 병행 학습해 비교하는 것도 방법이다. EXAONE은 성능이 매력적이지만, 실제로 API로 서비스화(상업적 재배포)할 계획이라면 라이선스 조항부터 법무적으로 확인한 뒤 후보에 넣어야 한다.

모델 크기는 8~12B급을 기준으로 잡았다. 요약·인사이트 추출은 추론(reasoning)보다 지시 이행·문체 일관성이 중요한 태스크라, 30B급 이상으로 올리는 것보다 이 구간을 잘 파인튜닝하는 쪽이 비용 대비 효율이 높다. 14B 이상은 코랩 무료 티어 QLoRA로는 세션을 여러 번 나눠도 부담이 커 1차 후보에서 제외했다.

### 1.1 코랩 무료 티어(T4, VRAM 약 15GB) QLoRA 예상 학습 시간

전제: 4bit QLoRA, gradient checkpointing 필수(안 켜면 활성화 메모리로 OOM), T4는 bf16 미지원이라 compute dtype은 fp16, batch size 1 + gradient accumulation으로 유효 배치 16, 시퀀스 길이 약 1024토큰. 무료 티어는 세션당 최대 12시간·유휴 90분 컷·주간 GPU 사용량 15~30시간으로 변동적이라, 아래 시간이 넘으면 체크포인트 저장 후 세션을 나눠 이어 학습하는 걸 기본 전제로 잡는다.

| 모델 | 스텝당 시간(대략) | 1 epoch (3,000건 기준) | 2~3 epoch 총 시간 | 비고 |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 4~6초 | 3.5~5시간 | 7~15시간 | 세션 1~2회로 커버 가능 |
| Gemma 4 12B / Mistral NeMo 12B | 6~9초 | 5~7.5시간 | 10~22시간 | 세션 2~3회 필요, OOM 시 시퀀스를 512까지 축소 |

데이터가 1,000건이면 위 수치를 대략 1/3로, 10,000건이면 3배로 보면 된다. `Trainer`의 `save_strategy="steps"`로 구글 드라이브에 짧은 주기로 체크포인트를 저장하고, 다음 세션에서 `resume_from_checkpoint`로 이어 붙이는 방식을 권장한다.

## 2. 파인튜닝 방식 (LoRA/QLoRA)

학습 프레임워크는 Hugging Face `transformers` + `peft` + `trl`(SFTTrainer) 조합을 기본으로 하고, 학습 속도/메모리 효율이 더 필요하면 Unsloth로 교체 가능하게 설계한다(같은 LoRA adapter 포맷과 호환).

권장 하이퍼파라미터 초안은 다음과 같다.

| 항목 | 값 | 비고 |
| --- | --- | --- |
| 양자화 | QLoRA 4bit (nf4, bf16 compute — 단 코랩 무료 T4는 bf16 미지원이라 fp16 compute로 대체) | 8~12B 모델을 24GB급 GPU 1장, 또는 코랩 무료 T4(15GB)에서도 학습 가능(1.1절 예상 시간 참고) |
| LoRA rank (r) | 16 | 데이터가 수천 건 수준이면 16~32면 충분 |
| LoRA alpha | 32 | 통상 r의 2배 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | attention+MLP 전체에 적용해야 요약 스타일 학습이 잘 붙는다 |
| learning rate | 1e-4 ~ 2e-4 (cosine decay) | |
| epoch | 2~3 | 데이터가 적을수록 과적합 주의 |
| effective batch size | 16~32 (gradient accumulation 활용) | |

학습 입력 포맷은 지금 `tools/summarize.py`/`tools/insight.py`의 프롬프트·출력 스키마(JSON)를 그대로 학습 타깃으로 삼는다. 즉 "기사 원문 → summarize 프롬프트 → JSON 요약" 쌍과 "요약 묶음 → insight 프롬프트 → 인사이트/시사점 JSON" 쌍을 각각 별도 LoRA로 학습하거나, 하나의 adapter에 두 태스크를 함께 학습(멀티태스크)할 수 있다. 처음에는 요약 태스크 하나만 떼어서 검증하고, 안정화되면 인사이트 추출로 확장하는 순서를 권장한다.

## 3. 학습 데이터 확보 방안

두 축으로 데이터를 모은다.

### 3.1 자체 파이프라인 데이터 (1차, 가장 중요)

BrieFYI는 이미 `digests` 테이블에 `summary_json`/`insight_json`을 계속 쌓고 있다. 즉 (원문 기사, Claude가 만든 요약/인사이트) 쌍이 파이프라인을 돌릴수록 자동으로 축적된다. 이건 실제 서비스 톤·포맷에 정확히 맞는 데이터라, 공개 데이터셋보다 파인튜닝 효과가 크다. 지식 증류(distillation) 관점에서 Claude를 교사 모델, 새 오픈모델을 학생 모델로 보는 구조다.

**단, 이 방식은 Anthropic 이용약관을 반드시 짚고 가야 한다.** Anthropic 상업 약관은 Claude의 output을 "Anthropic 모델과 경쟁하는 모델"을 학습시키는 데 쓰는 것은 금지하지만, "자체 제품에 통합되는 특화된 도구·분류기"를 만드는 용도는 허용한다고 명시하고 있다. BrieFYI의 요약 모델은 범용 LLM API로 되팔 게 아니라 이 파이프라인 내부의 좁은 요약 기능에만 쓰는 것이므로 후자에 해당할 가능성이 높지만, 판단이 애매한 경계 사안이라 실제 상업적으로 크게 굴릴 계획이면 한 번 더 약관을 검토하거나 Anthropic에 직접 문의해보는 걸 권한다. 리스크를 낮추려면 Claude 생성 데이터에만 100% 의존하지 말고 아래 3.2 공개 데이터와 섞어서 쓰는 것이 안전하다.

수집 방법은 간단하다. `digests` 테이블에서 `summary_json`, `insight_json`과 그 시점 `raw_articles`를 조인해 (원문, 출력) 쌍을 주기적으로 export하는 스크립트를 하나 만들면 된다. 파이프라인을 계속 돌릴수록 데이터가 자동으로 늘어나는 구조라, 초기 수백 건에서 시작해 몇 주 안에 수천 건 규모로 키울 수 있다.

### 3.2 공개 데이터셋 (2차, 기반 다지기용)

| 데이터셋 | 규모 | 특징 |
| --- | --- | --- |
| AI Hub "문서요약 텍스트" | 원문 40만 건(신문기사 30만 포함), 추출/생성 요약 각 40만 | 한국어 뉴스·기고문·잡지·판결문 요약. 규모가 가장 크고 신뢰도 높음. 승인 절차 필요 |
| AI Hub "뉴스기사 기계독해 데이터" | - | 요약은 아니지만 질의응답 페어로 보조 학습(사실 파악 능력) 가능 |
| Dacon 뉴스 요약 경진대회 데이터 | 대회별 상이 | 무료, 규모는 작지만 빠르게 받을 수 있음 |
| XL-Sum 한국어 subset | 약 1.1만 건 (BBC 코리아) | 라이선스가 CC-BY-NC-SA라 비상업적 용도로만 사용 가능 — 상업 서비스면 제외하거나 검증용으로만 사용 |

전략은 AI Hub 문서요약 텍스트로 기본기(요약 형식·정보 압축 능력)를 잡고, 3.1의 자체 파이프라인 데이터로 파인튜닝 마지막 단계에서 BrieFYI 고유 톤·JSON 스키마에 맞추는 2단계 학습(base fine-tune → domain fine-tune)을 권장한다.

### 3.3 데이터 정제

원문 길이 편차가 큰 기사는 토큰 제한에 맞게 필터링하고, Claude 출력 중 `grounding_check`(검증 항목 #10, 아직 미구현이면 최소한 수동 샘플 검수)를 통과한 것만 학습 데이터로 채택해 "나쁜 요약을 나쁜 요약으로 학습"하는 것을 방지한다.

## 4. 검증(평가) 설계

한 가지 지표만으로는 요약 품질을 판단할 수 없으므로 4단계로 검증한다.

**1단계 — 자동 지표(개발 중 빠른 반복용).** ROUGE-1/2/L로 참조 요약과의 어휘 중복도를, BERTScore(다국어 모델 기반)로 의미적 유사도를 측정한다. 두 지표 모두 추상적 요약에서는 한계가 있어 참고용으로만 쓰고 합격선을 이 지표만으로 정하지는 않는다.

**2단계 — 구조적 검증(파이프라인 특화).** BrieFYI 출력 스키마를 그대로 지켰는지 자동 채점한다: JSON 파싱 성공률, 인사이트 개수가 3~5개 범위인지, 각 인사이트에 `source_url`이 실제로 붙어 있는지, 글자수 제약 준수율. 이건 rouge/bertscore보다 오히려 실사용에 더 중요한 합격 기준이다.

**3단계 — 사실 일치성(환각 검증).** 요약이 원문에 없는 내용을 만들어내는지 확인한다. AlignScore/SummaC류의 오픈소스 사실성 채점 모델을 쓰거나, 더 간단하게는 Claude를 "판정자"로 세워 "이 요약 문장이 원문에 근거하는가"를 항목별로 채점시키는 LLM-as-judge 방식을 쓴다. 학습에 쓰인 교사 모델(Claude)을 채점자로도 쓰는 게 이상해 보일 수 있지만, 여기서는 새 모델을 훈련시키는 게 아니라 "원문 대비 사실성"이라는 객관적 기준을 판정하는 것이므로 문제없다.

**4단계 — 사람 평가 + 프로덕션 섀도우 테스트.** 골든 테스트셋(대표 기사 50~100건)에 대해 Claude 요약과 새 모델 요약을 블라인드로 나란히 놓고 팀이 직접 승패를 매기는 pairwise 비교(win-rate)를 한다. 자동 지표를 통과해도 이 단계에서 최종 결정한다. 이후 실제 배포 전에는 파이프라인에서 두 모델을 동시에 돌리되(섀도우 모드) 새 모델의 출력은 사용자에게 보내지 않고 로그만 남겨 며칠~몇 주간 비교하고, 이상 없으면 트래픽의 일부(예: 10%)부터 점진적으로 전환한다. `grounding_check` 실패율이나 사용자 피드백(예: 이메일 옵트아웃률)이 기준치를 넘으면 자동으로 Claude로 롤백하는 회로도 함께 설계해두는 것이 안전하다.

이 4단계를 새 학습 버전을 낼 때마다 반복하는 회귀 테스트 세트로 고정해두면, 데이터나 하이퍼파라미터를 바꿀 때마다 품질이 실제로 좋아졌는지 나빠졌는지 일관되게 비교할 수 있다.

## 5. Hugging Face 배포

베이스 모델은 그대로 두고 LoRA adapter만 별도 레포로 Hugging Face Hub에 올린다(용량이 작아 관리·버전 롤백이 쉽다). 배포 방식은 세 가지를 검토할 수 있다.

Hugging Face **Inference Endpoints**는 전용 GPU 인스턴스를 관리형으로 띄워주고 오토스케일까지 지원해 운영 부담이 가장 적다. 대신 인스턴스 가동 시간만큼 과금되므로 트래픽이 적으면 상대적으로 비싸다. 자체 서버나 클라우드 VM에 **vLLM/TGI**로 직접 띄우는 방식은 LoRA adapter를 여러 개 동시에 로드할 수 있고(멀티 adapter serving) 비용을 더 세밀하게 통제할 수 있지만, 운영·모니터링을 직접 해야 한다. 트래픽이 일정치 않다면 서버리스形 Inference Providers(토큰당 과금)도 검토할 만하지만 콜드스타트 지연이 있을 수 있다.

BrieFYI 쪽 통합은 `tools/summarize.py`/`tools/insight.py`의 `call_llm`을 provider 추상화로 바꿔, 환경변수(`SUMMARIZER_PROVIDER=anthropic|hf`)로 Claude와 새 엔드포인트를 스위치할 수 있게 하는 정도의 변경이면 충분하다. 처음에는 `hf` 모드를 섀도우로만 돌리다가, 검증을 통과하면 기본값을 바꾸는 순서를 권장한다.

## 6. 데이터·학습 코드 통일을 위한 기능 구조 (팀원 제안과 통합)

이 절은 원래 제안(다중 소스 데이터 통일 + config 기반 학습 + 4단계 평가)에, 팀원이 별도로 제안한 `src/exaone_summarize/` 구조(실제 패키징, 학습 정합성 디테일, 테스트)를 병합한 최종안이다. 골격은 팀원 제안의 `src/` 레이아웃·테스트·환경점검·정합성 디테일을 그대로 채택하고, 거기에 빠져 있던 다중 소스 데이터 계층과 4단계 평가를 편입했다. 패키지명은 특정 모델에 종속되지 않도록 `exaone_summarize` 대신 `summarize_ft`로 일반화했다(EXAONE만 쓰기로 확정되면 다시 좁혀도 됨).

### 6.1 디렉터리 구조

```
finetune/
├── configs/
│   ├── qlora_qwen3-8b.yaml       # 기본 추천 모델
│   ├── qlora_gemma4-12b.yaml      # 성능 우선 대안
│   ├── qlora_exaone-7.8b.yaml      # 팀원이 검토 중이던 모델 (라이선스 확인 후 사용, 6.7절 참고)
│   ├── lora_bf16_7.8b.yaml          # 비양자화 버전(24GB+ GPU 있을 때)
│   └── smoke.yaml                     # 스모크 테스트용 초소형 설정(스텝 수·데이터 극소)
├── data/
│   └── sample/                          # 오프라인 스모크 테스트용 번들 샘플(8건/3건, API 키 없이 파이프라인 검증)
├── docs/
│   ├── USAGE.md
│   ├── ARCHITECTURE.md
│   └── WORKLOG.md                         # 실험별 요약 기록(6.6 registry의 사람이 읽는 버전)
├── scripts/
│   ├── setup.ps1 / setup.sh                 # 의존성 설치
│   ├── check_env.py                           # GPU/CUDA/패키지 버전 사전 점검
│   ├── prepare_data.py                         # sources/* 를 호출해 정제→분할까지 수행하는 오케스트레이션 스크립트
│   └── run_pipeline.ps1 / run_pipeline.sh        # prepare → train → eval을 순서대로 실행
├── src/summarize_ft/
│   ├── config.py                                  # 설정 스키마, --set 오버라이드, 정합성 검증 (팀원 제안 채택)
│   ├── schema.py                                   # 공통 학습 예제 스키마 (원 제안에서 편입, 6.2절)
│   ├── sources/                                     # 다중 데이터 소스 파서 (원 제안에서 편입, 6.2절)
│   │   ├── digests_export.py                          # digests+raw_articles 조인 → 공통 스키마
│   │   ├── aihub_loader.py                              # AI Hub 문서요약 텍스트 파서
│   │   └── dacon_loader.py                                # Dacon 대회 데이터 파서
│   ├── prompt.py                                    # chat template 구성, 문서 토큰 절단 (팀원 제안 채택)
│   ├── jsonl.py                                      # JSONL 입출력, torch 비의존 (팀원 제안 채택)
│   ├── data.py                                        # completion-only 마스킹, 동적 패딩 콜레이터 (팀원 제안 채택)
│   ├── modeling.py                                     # 4bit 양자화, target_modules 자동 감지, LoRA 부착 (팀원 제안 채택)
│   ├── train.py                                         # 학습 진입점
│   ├── infer.py                                          # 요약 생성
│   ├── evaluate.py                                        # ROUGE/BERTScore + 구조검증 + 사실성 + 사람평가 export (4단계로 확장, 6.5절)
│   ├── merge_lora.py                                       # 어댑터 병합 (팀원 제안 채택)
│   └── registry.py                                          # 실험 메타데이터 기록 (원 제안에서 편입, 6.6절)
└── tests/                                                    # 46개+ (모델 다운로드 불필요, 팀원 제안 채택)
```

### 6.2 공통 데이터 스키마와 다중 소스 파서 (원 제안에서 편입)

`src/summarize_ft/sources/*.py`는 자체 파이프라인(`digests` 테이블), AI Hub, Dacon 등 원본 포맷이 제각각인 데이터를 아래 공통 JSONL 스키마 하나로 변환하는 역할만 한다. 이후 단계(`data.py`의 마스킹/콜레이팅, `train.py`)는 데이터가 어디서 왔는지 전혀 신경 쓸 필요가 없다. `scripts/prepare_data.py`가 이 소스들을 순서대로 호출해 정제·분할까지 끝낸 파일을 `data/processed/`에 만든다.

```json
{
  "id": "uuid",
  "task": "summarize",
  "source": "digest_pipeline",
  "input": {
    "article_title": "...",
    "article_text": "...",
    "prompt_template": "summarize_v1"
  },
  "output": { "topic_title": "...", "summary": "...", "source_urls": ["..."] },
  "meta": {
    "created_at": "2026-08-15T00:00:00Z",
    "teacher_model": "claude-sonnet-4-5",
    "quality_flag": "verified"
  }
}
```

`task`는 `summarize` 또는 `insight`이고 `output`은 `tools/summarize.py`/`tools/insight.py`가 이미 쓰는 JSON 스키마를 그대로 재사용해, 학습 데이터와 프로덕션 출력 형식이 항상 일치하게 한다.

### 6.3 설정 기반 학습 진입점 (팀원 제안 채택 + target_modules 자동 감지)

`config.py`가 YAML을 로드하고 `--set train.learning_rate=1e-4` 같은 CLI 오버라이드와 정합성 검증(필수 필드 누락, 타입 체크)까지 처리한다. 원래 제안에서는 `target_modules`를 모델마다 config에 수동으로 적어야 했는데, `modeling.py`가 베이스 모델의 `nn.Linear` 레이어를 스캔해 자동으로 채우도록 팀원 제안을 그대로 채택했다 — 이러면 새 베이스 모델을 추가할 때 config에서 그 줄 자체를 지워도 된다.

```yaml
# configs/qlora_qwen3-8b.yaml
base_model: Qwen/Qwen3-8B-Instruct
task: summarize
quantization: qlora_4bit
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  # target_modules는 생략 가능 — modeling.py가 자동 감지. 특정 레이어만 쓰고 싶을 때만 명시.
train:
  learning_rate: 2e-4
  epochs: 3
  effective_batch_size: 16
  max_seq_len: 1024
data:
  train_path: data/processed/summarize_train.jsonl
  val_path: data/processed/summarize_val.jsonl
  data_version: v1
output_dir: runs/qwen3-8b-summarize-v1
```

```bash
python -m summarize_ft.train --config finetune/configs/qlora_qwen3-8b.yaml
python -m summarize_ft.train --config finetune/configs/qlora_gemma4-12b.yaml
python -m summarize_ft.train --config finetune/configs/smoke.yaml --set train.epochs=1   # 스모크 테스트
```

`prompt.py`가 모델별 chat template 적용과 긴 기사의 토큰 절단을 전담하고, `data.py`가 completion-only 마스킹(프롬프트 부분은 loss 계산에서 제외)과 동적 패딩 콜레이터를 제공한다. 둘 다 원래 제안엔 없던, 팀원 제안에서 가져온 실제 학습 품질에 중요한 디테일이다.

### 6.4 스모크 테스트와 온보딩 (팀원 제안 채택)

`data/sample/`에 8건/3건짜리 번들 샘플을 넣어두고, `configs/smoke.yaml` + `python -m summarize_ft.train --config configs/smoke.yaml`로 GPU가 없거나 실제 데이터/API 키가 아직 없어도 파이프라인이 끝까지 도는지 몇 분 안에 확인할 수 있게 한다. `scripts/check_env.py`는 CUDA·GPU·필수 패키지 버전을 미리 점검해 "학습 몇 시간 돌리다 중간에 환경 문제로 실패"하는 걸 막는다. `tests/`(46개+)는 모델 다운로드 없이 `config.py`/`schema.py`/`data.py`의 로직만 검증한다.

### 6.5 평가 실행 — 4단계로 확장 (원 제안 + 팀원 제안 결합)

`evaluate.py`는 팀원 제안대로 하나의 스크립트로 통일하되, ROUGE만이 아니라 원래 설계한 4단계를 전부 포함하도록 확장한다.

```bash
python -m summarize_ft.evaluate --checkpoint runs/qwen3-8b-summarize-v1/adapter --testset data/golden/testset_v1.jsonl
python -m summarize_ft.evaluate --checkpoint runs/gemma4-12b-summarize-v1/adapter --testset data/golden/testset_v1.jsonl
```

내부적으로 1단계(ROUGE/BERTScore) → 2단계(JSON 스키마·인사이트 개수 등 구조 검증) → 3단계(grounding 기반 사실성) → 4단계(사람평가용 pairwise 파일 export) 순으로 실행하고 리포트를 만든다. 체크포인트만 바꿔 끼우면 되므로 Qwen3-8B와 Gemma 4 12B를 같은 골든셋으로 나란히 비교할 수 있다.

### 6.6 실험 레지스트리 (원 제안에서 편입)

`registry.py`가 학습 실행마다 config 해시, 데이터 버전, 4단계 평가 점수를 로그(JSON Lines, 필요하면 Postgres `finetune_runs` 테이블)로 남긴다. `docs/WORKLOG.md`는 이 로그를 사람이 읽기 좋게 요약하는 용도로 남겨두되, 실제 비교·검색은 `registry.py`의 구조화된 로그를 기준으로 한다.

### 6.7 확인이 필요한 부분

`scripts/prepare_data.py`가 실제로 여러 소스를 다 지원하는지, 아니면 단일 포맷을 전제로 짜여 있는지는 코드를 봐야 확정할 수 있다. 다중 소스라면 그대로 `sources/*.py` 호출부만 추가하면 되고, 단일 포맷이라면 `sources/` 계층을 새로 끼워 넣는 리팩터링이 필요하다. 또한 패키지명·config가 EXAONE(7.8B)을 가리키고 있는데, 1절에서 짚었듯 EXAONE은 자체 라이선스라 상업적 재배포 조건을 먼저 확인해야 한다 — 이미 검토됐다면 문제없지만, 아직이라면 Qwen3-8B로 먼저 검증하고 EXAONE은 후보로만 남겨두는 게 안전하다.

## 7. 로드맵

1단계는 `scripts/check_env.py`로 환경을 점검하고 `configs/smoke.yaml` + `data/sample/`로 파이프라인이 끝까지 도는지 스모크 테스트부터 통과시킨다. 2단계는 `src/summarize_ft/sources/digests_export.py`로 `digests` 테이블에서 학습 데이터를 공통 스키마로 export하고, 동시에 AI Hub 문서요약 데이터를 신청·확보해 같은 스키마로 변환한다(`scripts/prepare_data.py`). 3단계는 `configs/qlora_qwen3-8b.yaml`로 QLoRA 1차 학습을 돌리고 2단계 검증(자동 지표+구조 검증)까지 통과시킨다. 여유가 되면 `configs/qlora_gemma4-12b.yaml`로 같은 데이터를 병행 학습해 두 모델을 비교하고, EXAONE 후보는 라이선스 확인이 끝나면 `configs/qlora_exaone-7.8b.yaml`로 같이 돌려본다. 4단계는 골든 테스트셋으로 `evaluate.py`를 돌려 사람 평가까지 진행하고, 가장 나은 조합을 `merge_lora.py`로 병합하거나 adapter 그대로 Hugging Face에 배포해 섀도우 모드로 붙인다. 5단계는 섀도우 결과가 안정적이면 트래픽 일부를 점진 전환하고, 비용/품질 트레이드오프를 보며 Claude와의 비중을 조정한다.

## Sources

- [The Best Open Source LLMs for Summarization in 2026](https://www.siliconflow.com/articles/en/best-open-source-llms-for-summarization)
- [한국어를 지원하는 오픈 모델과 토크나이저 비용](https://www.youngju.dev/blog/ai/2026-08-12-huggingface-korean-models)
- [국내 LLM 모델들의 현황과 비교 - MSAP](https://www.msap.ai/blog-home/blog/korea-llm/)
- [AI-Hub 문서요약 텍스트](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=97)
- [LoRA (Low-Rank Adaptation) · Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/main/conceptual/lora)
- [Can I use my Outputs to train an AI model? | Claude Help Center](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model)
- [Anthropic's terms typically restrict training competing AI models with their outputs (Hacker News discussion)](https://news.ycombinator.com/item?id=44429697)
- [Qwen 3.6 Series: Alibaba's Open-Source LLM (Apache 2.0)](https://aimlapi.com/blog/qwen-3-6-series-alibabas-open-source-llm-revolution-in-2026)
- [Gemma 4: Expanding the Gemmaverse with Apache 2.0 | Google Open Source Blog](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html)
- ['Open' AI model licenses often carry concerning restrictions (Gemma 3 라이선스 이슈 배경) | TechCrunch](https://techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions/)
- [Google Colab Free Tier Limits (2026): GPU, Runtime & Pricing](https://joshthompson.co.uk/ai/google-colab-2026-guide-free-compute-automations-pro-tips/)
- [Google Colab GPU: free access, limits, and alternatives | Hivenet](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)
