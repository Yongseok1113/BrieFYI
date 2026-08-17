"""실제 뉴스 RAG에서 사용하는 Category/Domain 고정 taxonomy."""

TOPIC_TAXONOMY_VERSION = "business_tech_v1"

CATEGORY_LABELS = ("경제", "기술", "금융", "산업", "기타")

DOMAIN_LABELS = (
    "반도체",
    "AI",
    "배터리",
    "바이오",
    "자동차",
    "부동산",
    "증시",
    "통화정책",
    "스타트업",
    "클라우드",
    "로봇",
    "에너지",
    "커머스",
    "게임",
    "기타",
)

DOMAIN_CLASSIFICATION_THRESHOLD = 0.4
MAX_ENTITIES = 3

