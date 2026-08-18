"""RAG 4-Layer(Category/Domain/Entity/Event)의 고정 taxonomy와 추출 설정.

Category/Domain/Entity와 Event는 같은 GLiNER2 model 하나(`GLINER2_MODEL_NAME`)로 뽑으므로
두 taxonomy를 한 파일에 둔다. Event 쪽 version 값들은 `article_event_index_status`에
저장돼 재처리 여부 판단에 쓰인다.
"""

# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------

GLINER2_MODEL_NAME = "fastino/gliner2-multi-v1"

# GLiNER2 encoder의 512-token 한계에서 schema 입력 몫을 남긴 window 설정.
# BGE chunk(500 token / 50 token overlap)와는 tokenizer도 단위도 다르다.
MAX_TOKENS_PER_WINDOW = 400
OVERLAP_WORDS = 40

ENTITY_TYPES = {
    "company": "company or business",
    "organization": "government body, institution, or organization",
    "person": "a named individual",
    "product": "a named product or service",
}

# ---------------------------------------------------------------------------
# Category / Domain / Entity (article-level metadata)
# ---------------------------------------------------------------------------

# 저장할 컬럼이 아직 없어 추출 결과에는 싣지 않는다. taxonomy를 바꿀 때 이 값을 올리고
# 어디까지 재인덱싱해야 하는지 판단하는 기준으로만 쓴다.
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

# ---------------------------------------------------------------------------
# 구조화 Event
# ---------------------------------------------------------------------------

EVENT_TAXONOMY_VERSION = "ai_tech_v1"
EVENT_EXTRACTION_VERSION = "gliner2_event_v1"

RELATION_THRESHOLD = 0.5

# GLiNER2의 head/tail을 저장용 argument role로 변환한다.
RELATION_ROLES = {
    "released": ("releaser", "released_item"),
    "developed": ("developer", "developed_item"),
    "invested_in": ("investor", "investee"),
    "acquired": ("acquirer", "acquired_entity"),
    "partnered_with": ("partner_a", "partner_b"),
    "integrated_with": ("integrator", "integrated_item"),
    "supplies_to": ("supplier", "customer"),
    "sued": ("plaintiff", "defendant"),
    "appointed_at": ("appointee", "organization"),
    "resigned_from": ("person", "organization"),
}

SYMMETRIC_RELATIONS = frozenset({"partnered_with"})
