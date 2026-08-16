"""뉴스(및 향후 기술문서) 수집을 담당하는 에이전트.

지금은 GNews 하나뿐이라 fetch_news만 호출한다. 기술문서 수집(백로그 #11)이
추가되면 tools에 fetch_tech_docs를 등록하고, 이 안에서 "이번 실행에 어떤
소스를 돌릴지"를 판단하도록 확장하면 된다.
"""
from .base import Agent


class CollectorAgent(Agent):
    def run(self, state: dict) -> dict:
        articles = self.tools["fetch_news"](
            state["keyword"], state["lookback_days"], state["max_results"]
        )
        return {"raw_articles": articles}
