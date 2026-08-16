"""라우팅 판단을 전담하는 에이전트.

지금 graph/pipeline.py는 조건 분기 없는 고정 파이프라인이라, 이 클래스의 판단
메서드는 아직 그래프에 실제로 연결되어 있지 않다. 나중에 조건부 엣지
(add_conditional_edges)를 그래프에 추가할 때 "어디로 갈지"의 판단 로직을
이 클래스 한 곳에 모아두는 게 목적이다. 예를 들어 신규 기사가 없을 때 요약
단계를 건너뛰는 이벤트 리스너 동작이나, 검증 실패 시 요약 단계로 되돌리는
로직이 여기 추가될 수 있다.
"""
from .base import Agent


class OrchestratorAgent(Agent):
    def decide_after_store(self, state: dict) -> str:
        """store_raw 다음 어디로 갈지 판단한다.

        신규 저장된 기사가 없으면 뒤 단계(요약/인사이트/발송)를 건너뛰고 바로
        종료하자는 판단이다. 그래프에 연결하려면
        graph.add_conditional_edges("store_raw", agents["orchestrator"].decide_after_store,
        {"summarize": "summarize", "end": END}) 형태로 store_raw -> summarize 고정
        엣지를 조건부 엣지로 바꿔야 한다.
        """
        return "summarize" if state.get("inserted_count", 0) > 0 else "end"
