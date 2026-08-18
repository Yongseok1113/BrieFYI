"""전역 설정 로더. 레포 루트 config.py와 같은 패턴이지만 코드는 완전히 독립적이다
(별도 컨테이너로 빌드되므로 레포 루트 코드를 import하지 않는다).

.env는 레포 루트 하나로 통합 관리한다 — data_pipeline/에는 별도 .env를 두지 않고,
아래에서 레포 루트의 .env를 명시적으로 가리켜서 읽는다.
"""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# data_pipeline/src/data_pipeline/config.py -> parents[3] == 레포 루트 (로컬 실행 기준).
# 컨테이너 안에서는 이 경로에 .env가 없지만(docker-compose.yml이 env_file로 이미
# 프로세스 환경변수에 주입해줌) load_dotenv는 파일이 없으면 조용히 넘어가므로 문제없다.
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")


class Config:
    # 메인 앱과 같은 Postgres 인스턴스를 공유한다 (raw_articles/digests를 같이 보기 위함).
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "briefyi")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "briefyi")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "briefyi")
    DATABASE_URL = os.getenv("DATABASE_URL") or (
        f"postgresql://{quote_plus(POSTGRES_USER)}:{quote_plus(POSTGRES_PASSWORD)}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    # GNews (구조화 소스)
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
    GNEWS_BASE_URL = "https://gnews.io/api/v4/search"
    NEWS_KEYWORD = os.getenv("NEWS_KEYWORD", "AI")
    NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "1"))
    NEWS_MAX_RESULTS = int(os.getenv("NEWS_MAX_RESULTS", "10"))

    # LLM (변형2 enrich, 변형3 normalize fallback). 파인튜닝 안 한 기본 모델을
    # 프롬프트 엔지니어링만으로 호출하는 게 기본 전제다 — Claude 비용/약관을 피하기 위함.
    #
    # 기본값을 Groq로 전환했다: HF Inference Providers는 무료 계정 기준 월 $0.10
    # 크레딧뿐이고 그마저 카드 등록이 있어야 라우팅이 열려서(등록 안 하면 provider를
    # 명시해도 model_not_supported로 거부됨) 실제 파이프라인 운영엔 못 쓴다. Groq
    # 무료 티어는 카드 등록 없이 30 req/min·6000 tokens/min·14400 req/day를 준다 —
    # 이 파이프라인 규모엔 이쪽이 실질적으로 더 "무료"다.
    DATA_PIPELINE_LLM_PROVIDER = os.getenv("DATA_PIPELINE_LLM_PROVIDER", "groq")  # "groq" | "hf" | "anthropic"

    # Groq (기본값)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    DATA_PIPELINE_GROQ_MODEL = os.getenv("DATA_PIPELINE_GROQ_MODEL", "llama-3.3-70b-versatile")

    # HF (DATA_PIPELINE_LLM_PROVIDER=hf로 바꿨을 때만 쓰는 대체 경로 — 위 이유로 기본값 아님)
    # HF_API_TOKEN은 메인 앱(tools/hf_llm_client.py)과 공유해서 쓰지만, 모델 ID는
    # 반드시 분리한다 — 메인 앱의 HF_MODEL_ID는 "직접 파인튜닝해 push한 모델"을
    # 가리키고, 여기는 "학습 데이터를 만들 때 쓰는 파인튜닝 안 한 베이스 모델"을
    # 가리켜서 목적이 다르다. 같은 이름을 쓰면 둘 중 하나가 의도치 않게 덮어써진다.
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    DATA_PIPELINE_HF_MODEL_ID = os.getenv("DATA_PIPELINE_HF_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
    # HF의 Inference Providers 자동 라우팅("auto")이 계정에 활성화 안 된 provider로
    # 잘못 붙으려다 model_not_supported로 실패하는 걸 겪어서, provider를 명시적으로
    # 고정한다. https://huggingface.co/settings/inference-providers 에서 활성화한
    # provider와 이 값이 일치해야 한다.
    DATA_PIPELINE_HF_PROVIDER = os.getenv("DATA_PIPELINE_HF_PROVIDER", "featherless-ai")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    # 하위 호환용 별칭 (llm_client.py 등 내부 코드는 짧은 이름을 쓴다)
    LLM_PROVIDER = DATA_PIPELINE_LLM_PROVIDER
    GROQ_MODEL = DATA_PIPELINE_GROQ_MODEL
    HF_MODEL_ID = DATA_PIPELINE_HF_MODEL_ID

    # 요청 제한 (7절). Groq 무료 티어(30 req/min)에 안전 마진을 두고 기본값을 잡았다 —
    # 다른 provider로 바꾸면 .env에서 재조정할 것.
    RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "25"))
    RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))

    # 키워드 추출 (변형1, 로컬 전용 — API 호출 아님)
    KEYWORD_TOP_N = int(os.getenv("KEYWORD_TOP_N", "5"))
    KEYWORD_EMBEDDING_MODEL = os.getenv(
        "KEYWORD_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 통합 단어 테이블 (4절)
    SYNONYM_CLUSTER_THRESHOLD = float(os.getenv("SYNONYM_CLUSTER_THRESHOLD", "0.82"))
    SYNONYM_FUZZY_THRESHOLD = float(os.getenv("SYNONYM_FUZZY_THRESHOLD", "0.85"))
    SYNONYM_TABLE_VERSION = os.getenv("SYNONYM_TABLE_VERSION", "v1")

    # 단계별 배치 크기
    BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "20"))


config = Config()
