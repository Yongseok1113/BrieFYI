"""공통 Agent 베이스.

각 에이전트는 자신이 실제로 쓸 수 있는 tool만 `tools` 딕셔너리로 주입받는다.
예를 들어 CollectorAgent에는 fetch_news만 주고 send_email은 주지 않는 식으로,
"이 에이전트가 어떤 tool에 접근 가능한지"를 코드 레벨에서 제한한다
(agent-management-structure.md 3절 참고).
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Agent:
    name: str
    tools: dict[str, Callable] = field(default_factory=dict)
