"""LangGraph 기반 오케스트레이션 (구현 항목 #7).

지금 단계는 '고정 파이프라인'이므로 조건 분기 없이
fetch -> store -> summarize -> insight -> format -> send 순서로만 흐른다.
각 노드는 agents/ 패키지의 해당 역할 에이전트에 위임한다(agents/registry.py 참고).
추후 조건부 엣지(add_conditional_edges)를 추가할 때는
OrchestratorAgent.decide_after_store 같은 판단 메서드를 연결하면 된다.
"""
from datetime import date
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.registry import build_agents
from db.db import insert_articles
from tools.email_format import format_email

agents = build_agents()


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
    return agents["collector"].run(state)


def store_raw_node(state: PipelineState) -> dict:
    inserted = insert_articles(state["digest_date"], state["raw_articles"])
    return {"inserted_count": inserted}


def summarize_node(state: PipelineState) -> dict:
    return agents["summarizer"].run_summarize(state)


def insight_node(state: PipelineState) -> dict:
    return agents["summarizer"].run_insight(state)


def format_email_node(state: PipelineState) -> dict:
    html = format_email(state["digest_date"], state["keyword"], state["summaries"], state["insight"])
    return {"email_html": html}


def send_email_node(state: PipelineState) -> dict:
    return agents["distributor"].run(state)


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
