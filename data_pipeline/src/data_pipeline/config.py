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
    # HF 무료 서버리스 API로 호출하는 것이 기본 전제다 — Claude 비용/약관을 피하기 위함.
    # HF_API_TOKEN은 메인 앱(tools/hf_llm_client.py)과 공유해서 쓰지만, 모델 ID는
    # 반드시 분리한다 — 메인 앱의 HF_MODEL_ID는 "직접 파인튜닝해 push한 모델"을
    # 가리키고, 여기는 "학습 데이터를 만들 때 쓰는 파인튜닝 안 한 베이스 모델"을
    # 가리켜서 목적이 다르다. 같은 이름을 쓰면 둘 중 하나가 의도치 않게 덮어써진다.
    DATA_PIPELINE_LLM_PROVIDER = os.getenv("DATA_PIPELINE_LLM_PROVIDER", "hf")  # "hf" | "anthropic"
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    # Qwen2.5-14B-Instruct: Featherless provider로 실제 서빙 확인된 모델 중 Qwen3-8B보다
    # 큰 무료 옵션. DeepSeek-V4-Flash 등 더 강한 모델도 있지만 :novita 같은 특정 provider로
    # 붙이면 토큰당 소액 과금이 생겨(무료 아님) — 무료 유지가 목적이면 이 모델을 쓴다.
    # provider 배치는 라이브 상태라 바뀔 수 있다 — 안 되면 모델 페이지의 Inference
    # Providers 위젯에서 실제 서빙 중인 다른 모델로 바꿀 것.
    DATA_PIPELINE_HF_MODEL_ID = os.getenv("DATA_PIPELINE_HF_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
    # HF의 Inference Providers 자동 라우팅("auto")이 계정에 활성화 안 된 provider로
    # 잘못 붙으려다 model_not_supported로 실패하는 걸 겪어서, provider를 명시적으로
    # 고정한다. https://huggingface.co/settings/inference-providers 에서 활성화한
    # provider와 이 값이 일치해야 한다 (지금은 Featherless AI 활성화 확인됨).
    DATA_PIPELINE_HF_PROVIDER = os.getenv("DATA_PIPELINE_HF_PROVIDER", "featherless-ai")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    # 하위 호환용 별칭 (llm_client.py 등 내부 코드는 짧은 이름을 쓴다)
    LLM_PROVIDER = DATA_PIPELINE_LLM_PROVIDER
    HF_MODEL_ID = DATA_PIPELINE_HF_MODEL_ID

    # 요청 제한 (7절). HF 무료 티어는 계정/모델별로 변동이 있어 파일럿 실행 중 조정 필요.
    RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "250"))
    RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
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
