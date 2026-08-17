"""GLiNER2 구조화 Event의 고정 taxonomy와 추출 설정."""

MODEL_NAME = "fastino/gliner2-multi-v1"
TAXONOMY_VERSION = "ai_tech_v1"
EXTRACTION_VERSION = "gliner2_event_v1"

RELATION_THRESHOLD = 0.5
MAX_TOKENS_PER_WINDOW = 400
OVERLAP_WORDS = 40

ENTITY_TYPES = {
    "company": "company or business",
    "organization": "government body, institution, or organization",
    "person": "a named individual",
    "product": "a named product or service",
}

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
