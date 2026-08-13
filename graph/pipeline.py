"""LangGraph 기반 오케스트레이션 (구현 항목 #7).

지금 단계는 '고정 파이프라인'이므로 조건 분기 없이
fetch -> store -> summarize -> insight -> format -> send 순서로만 흐른다.
추후 에이전트화(재시도 판단, 소스 선택 등)할 때 이 그래프에
조건부 엣지(add_conditional_edges)를 추가하면 된다.
"""
from datetime import date
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from db.db import insert_articles, log_send, save_digest
from tools.email_format import format_email
from tools.email_send import send_email
from tools.insight import extract_insights
from tools.news_fetch import fetch_news
from tools.summarize import summarize_articles


class PipelineState(TypedDict):
    keyword: str
    lookback_days: int
    max_results: int
    digest_date: str
    raw_articles: list[dict]
    inserted_count: int
    summaries: list[dict]
    insight: dict
    email_html: str
    digest_id: int
    send_result: Optional[dict]
    error: Optional[str]


def fetch_news_node(state: PipelineState) -> dict:
    articles = fetch_news(state["keyword"], state["lookback_days"], state["max_results"])
    return {"raw_articles": articles}


def store_raw_node(state: PipelineState) -> dict:
    inserted = insert_articles(state["digest_date"], state["raw_articles"])
    return {"inserted_count": inserted}


def summarize_node(state: PipelineState) -> dict:
    summaries = summarize_articles(state["raw_articles"])
    return {"summaries": summaries}


def insight_node(state: PipelineState) -> dict:
    insight = extract_insights(state["summaries"])
    digest_id = save_digest(state["digest_date"], state["keyword"], state["summaries"], insight)
    return {"insight": insight, "digest_id": digest_id}


def format_email_node(state: PipelineState) -> dict:
    html = format_email(state["digest_date"], state["keyword"], state["summaries"], state["insight"])
    return {"email_html": html}


def send_email_node(state: PipelineState) -> dict:
    from config import config

    subject = f"[{state['digest_date']}] {state['keyword']} 뉴스·기술 다이제스트"
    try:
        result = send_email(subject, state["email_html"])
        log_send(state["digest_id"], "email", config.EMAIL_TO, "success")
        return {"send_result": result}
    except Exception as exc:  # noqa: BLE001 - MVP 단계는 넓게 잡고 로그만 남긴다
        log_send(state["digest_id"], "email", config.EMAIL_TO, "failed", str(exc))
        return {"error": str(exc)}


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("store_raw", store_raw_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("extract_insight", insight_node)
    graph.add_node("format_email", format_email_node)
    graph.add_node("send_email", send_email_node)

    graph.add_edge(START, "fetch_news")
    graph.add_edge("fetch_news", "store_raw")
    graph.add_edge("store_raw", "summarize")
    graph.add_edge("summarize", "extract_insight")
    graph.add_edge("extract_insight", "format_email")
    graph.add_edge("format_email", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()


def run_pipeline(keyword: str, lookback_days: int, max_results: int) -> PipelineState:
    app = build_graph()
    initial_state: PipelineState = {
        "keyword": keyword,
        "lookback_days": lookback_days,
        "max_results": max_results,
        "digest_date": date.today().isoformat(),
        "raw_articles": [],
        "inserted_count": 0,
        "summaries": [],
        "insight": {},
        "email_html": "",
        "digest_id": 0,
        "send_result": None,
        "error": None,
    }
    return app.invoke(initial_state)
