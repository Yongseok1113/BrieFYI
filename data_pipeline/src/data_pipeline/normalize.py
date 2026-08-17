"""변형3: 'enriched' 상태 raw_articles(정확히는 그 enrichment 행)의 원시값을
synonym_table 기준으로 정규화한다 (design doc 5절).

순서: exact match -> fuzzy match(rapidfuzz) -> (그래도 못 찾으면) LLM fallback.
LLM에는 synonym_table의 canonical_value 목록을 명시적으로 주고 그중에서 고르게 한다 —
자유 생성으로 두면 목록에 없는 카테고리를 계속 지어내는 문제가 생기기 때문이다.
매 항목마다 LLM을 부르면 요청 제한이 3배로 소모되므로, fuzzy match로 대부분을 처리하고
LLM은 정말 못 찾은 것만 호출한다.
"""
from __future__ import annotations

from . import db
from .config import config
from .llm_client import call_llm, parse_json_response

LLM_NORMALIZE_PROMPT = """다음 원시 분류값을 후보 목록 중 가장 알맞은 것 하나로 매핑하라.
후보 목록에 적절한 게 전혀 없으면 새 canonical 값을 제안해도 된다(가능하면 후보를 우선 사용).

# 원시값
{raw_value}

# 후보 목록
{candidates}

아래 JSON으로만 답하라: {{"canonical": "선택한 값"}}
"""


def _exact_match(raw_value: str, entries: list[dict]) -> str | None:
    for entry in entries:
        if raw_value == entry["canonical_value"] or raw_value in (entry["aliases"] or []):
            return entry["canonical_value"]
    return None


def _fuzzy_match(raw_value: str, entries: list[dict], threshold: float) -> str | None:
    from rapidfuzz import fuzz

    best_value, best_score = None, 0.0
    for entry in entries:
        candidates = [entry["canonical_value"], *(entry["aliases"] or [])]
        for candidate in candidates:
            score = fuzz.ratio(raw_value, candidate) / 100.0
            if score > best_score:
                best_value, best_score = entry["canonical_value"], score
    if best_score >= threshold:
        return best_value
    return None


def _llm_match(raw_value: str, entries: list[dict]) -> str:
    candidates = ", ".join(entry["canonical_value"] for entry in entries) or "(후보 없음)"
    raw = call_llm(
        "당신은 분류값을 정규화하는 도우미입니다.",
        LLM_NORMALIZE_PROMPT.format(raw_value=raw_value, candidates=candidates),
        max_tokens=200,
    )
    result = parse_json_response(raw)
    return result["canonical"]


def normalize_value(raw_value: str, entries: list[dict], *, fuzzy_threshold: float) -> tuple[str, str]:
    """(정규화된 값, 방법) 튜플을 반환한다. 방법: 'exact' | 'fuzzy' | 'llm'."""
    exact = _exact_match(raw_value, entries)
    if exact:
        return exact, "exact"

    fuzzy = _fuzzy_match(raw_value, entries, fuzzy_threshold)
    if fuzzy:
        return fuzzy, "fuzzy"

    canonical = _llm_match(raw_value, entries)
    return canonical, "llm"


def _normalize_list(raw_values: list[str], entries: list[dict], fuzzy_threshold: float) -> tuple[list[str], str]:
    """리스트형(domain/entity/event) 필드를 정규화한다. 항목 중 하나라도 llm을 썼으면
    전체 normalization_method를 'llm'으로 기록해 나중에 추적할 수 있게 한다."""
    results, methods = [], []
    for raw_value in raw_values or []:
        canonical, method = normalize_value(raw_value, entries, fuzzy_threshold=fuzzy_threshold)
        results.append(canonical)
        methods.append(method)
    overall = "llm" if "llm" in methods else ("fuzzy" if "fuzzy" in methods else "exact")
    return results, (overall if methods else "exact")


def run_normalize(limit: int) -> dict:
    rows = db.fetch_enriched_rows(limit)
    done, failed = 0, 0

    entries_by_dim = {dim: db.fetch_synonym_entries(dim) for dim in ("category", "domain", "entity", "event")}

    for row in rows:
        try:
            category, category_method = (
                normalize_value(row["raw_category"], entries_by_dim["category"], fuzzy_threshold=config.SYNONYM_FUZZY_THRESHOLD)
                if row["raw_category"]
                else (None, "exact")
            )
            domain, domain_method = _normalize_list(row["raw_domain"] or [], entries_by_dim["domain"], config.SYNONYM_FUZZY_THRESHOLD)
            entity, entity_method = _normalize_list(row["raw_entity"] or [], entries_by_dim["entity"], config.SYNONYM_FUZZY_THRESHOLD)
            event, event_method = _normalize_list(row["raw_event"] or [], entries_by_dim["event"], config.SYNONYM_FUZZY_THRESHOLD)

            methods = {category_method, domain_method, entity_method, event_method}
            overall_method = "llm" if "llm" in methods else ("fuzzy" if "fuzzy" in methods else "exact")

            db.update_enrichment_normalized(
                enrichment_id=row["enrichment_id"],
                raw_article_id=row["raw_article_id"],
                category=category,
                domain=domain,
                entity=entity,
                event=event,
                normalization_method=overall_method,
                synonym_table_version=config.SYNONYM_TABLE_VERSION,
            )
            done += 1
        except Exception as exc:  # noqa: BLE001
            db.mark_failed(row["raw_article_id"], "normalize", str(exc))
            failed += 1

    return {"stage": "normalize", "candidates": len(rows), "done": done, "failed": failed}
