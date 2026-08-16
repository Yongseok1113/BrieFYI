# 에이전트 관리 구조 설계

지금 `graph/pipeline.py`의 노드(`fetch_news_node`, `summarize_node`, `insight_node`, `send_email_node` 등)는 사실상 `tools/*.py`의 함수를 그대로 감싼 것뿐이고, "오케스트레이터/수집/요약/배포 에이전트"라는 개념은 설계 문서에만 있고 코드에는 별도 자리가 없다. `tools/llm_client.py`는 Anthropic 호출을 감싼 유틸일 뿐, 에이전트를 정의하거나 등록하는 곳이 아니다. 이 문서는 그 빈자리를 채우는 `agents/` 계층을 설계한다.

## 1. 역할 구분: tools vs agents

지금까지처럼 `tools/`는 그대로 둔다. `tools/*.py`는 외부 API 하나를 감싼 순수 함수(예: `fetch_news`, `send_email`)로, 상태를 모르고 판단도 하지 않는다. 그 위에 새로 두는 `agents/*.py`가 "이 단계에서 어떤 tool을, 어떤 설정으로, 언제 호출할지" 판단하는 계층이다. 즉 tools는 손발이고 agents는 그 손발을 어떻게 쓸지 정하는 역할 담당자다. 이렇게 나누면 나중에 "요약을 Claude로 할지 자체 LoRA 모델로 할지"(LoRA 설계 문서의 `SUMMARIZER_PROVIDER`) 같은 판단이 tools를 안 건드리고 agents 안에서만 바뀐다.

## 2. 디렉터리 구조

```
agents/
  __init__.py
  base.py            # Agent 추상 클래스
  registry.py          # 역할 이름 -> 에이전트 인스턴스 매핑, provider 스위치 조립
  orchestrator.py       # OrchestratorAgent: 다음 단계 판단(조건부 라우팅) 전담
  collector.py           # CollectorAgent: 뉴스/기술문서 수집 소스 선택+호출
  summarizer.py           # SummarizerAgent: 요약+인사이트, provider(anthropic|hf) 스위치
  distributor.py           # DistributorAgent: 채널별(email/discord) 발송 선택+호출
```

`graph/pipeline.py`는 그대로 두되, 노드 함수 내용이 `tools` 직접 호출에서 `agents["역할"].run(state)` 호출로 얇아진다.

## 3. 공통 Agent 인터페이스

```python
# agents/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Agent(ABC):
    name: str
    tools: dict[str, Callable] = field(default_factory=dict)

    @abstractmethod
    def run(self, state: dict) -> dict:
        """PipelineState를 받아 이 에이전트가 담당하는 부분만 갱신해 반환한다."""
```

모든 에이전트는 어떤 tool들에 접근 가능한지(`tools` 딕셔너리)를 생성 시점에 명시적으로 주입받는다. CollectorAgent에는 `fetch_news`만 주고 `send_email`은 주지 않는 식으로, 각 에이전트가 실제로 쓸 수 있는 tool 범위를 코드 레벨에서 제한해둔다. 이게 원래 설계 문서(3.3절)에서 말한 "수집 에이전트가 이 tool들을 호출한다"는 경계를 코드로 강제하는 부분이다.

## 4. 에이전트별 구현

### 4.1 OrchestratorAgent — 판단 전담

지금 "새 기사가 없으면 건너뛴다"(이벤트 리스너 논의에서 나온 조건부 엣지)처럼, 파이프라인 어디서 분기할지 결정하는 로직을 전부 여기 모은다. 그래프의 엣지 함수가 익명 람다로 흩어져 있지 않고 한 클래스에 모여 있어야, 나중에 "실시간 트리거일 땐 검증 단계를 건너뛴다" 같은 규칙이 늘어나도 어디를 고쳐야 할지 명확하다.

```python
# agents/orchestrator.py
from .base import Agent


class OrchestratorAgent(Agent):
    def run(self, state: dict) -> dict:
        return {}  # 오케스트레이터는 상태를 직접 바꾸지 않고 라우팅만 결정한다

    def decide_after_store(self, state: dict) -> str:
        """store_raw 다음 어디로 갈지. 신규 기사 없으면 바로 종료(이벤트 리스너 동작)."""
        return "summarize" if state.get("new_articles") else "end"

    def decide_after_verify(self, state: dict) -> str:
        """검증 실패 시 요약 단계로 되돌릴지 결정 (백로그 #10 도입 시 사용)."""
        return "summarize" if state.get("verify_failed") else "format_email"
```

### 4.2 CollectorAgent — 소스 선택

지금은 GNews 하나뿐이지만, 기술문서 수집(백로그 #11)이 추가되면 여기서 "이번 실행에 어떤 소스를 돌릴지"를 정한다.

```python
# agents/collector.py
from .base import Agent


class CollectorAgent(Agent):
    def run(self, state: dict) -> dict:
        articles = self.tools["fetch_news"](
            state["keyword"], state["lookback_days"], state["max_results"]
        )
        if "fetch_tech_docs" in self.tools and state.get("include_tech_docs"):
            articles += self.tools["fetch_tech_docs"](state.get("tech_sources", []))
        return {"raw_articles": articles}
```

### 4.3 SummarizerAgent — provider 스위치

LoRA 파인튜닝 설계 문서의 `SUMMARIZER_PROVIDER=anthropic|hf` 스위치가 들어갈 자리다. `tools/summarize.py`(Claude 호출)와 향후 추가될 `tools/summarize_hf.py`(Hugging Face 엔드포인트 호출)를 이 에이전트가 provider 값에 따라 골라 쓴다. 섀도우 모드(두 provider를 다 돌려서 비교)도 여기서 처리한다.

```python
# agents/summarizer.py
from .base import Agent


class SummarizerAgent(Agent):
    def __init__(self, name: str, tools: dict, provider: str = "anthropic", shadow: bool = False):
        super().__init__(name=name, tools=tools)
        self.provider = provider
        self.shadow = shadow  # True면 hf 결과도 같이 만들어 로그만 남기고 배포엔 안 씀

    def run(self, state: dict) -> dict:
        summarize_fn = self.tools["summarize_hf" if self.provider == "hf" else "summarize_claude"]
        summaries = summarize_fn(state["new_articles"])
        result = {"summaries": summaries}

        if self.shadow and "summarize_hf" in self.tools:
            result["shadow_summaries"] = self.tools["summarize_hf"](state["new_articles"])
        return result
```

### 4.4 DistributorAgent — 채널 선택

Discord 발송(백로그 #9)이 추가될 때도 오케스트레이터나 다른 에이전트는 안 건드리고, 여기 `tools` 딕셔너리에 `discord`를 하나 더 등록하고 `channels` 목록만 늘리면 된다.

```python
# agents/distributor.py
from .base import Agent


class DistributorAgent(Agent):
    def __init__(self, name: str, tools: dict, channels: list[str]):
        super().__init__(name=name, tools=tools)
        self.channels = channels  # 예: ["email"] 또는 ["email", "discord"]

    def run(self, state: dict) -> dict:
        results = {}
        for channel in self.channels:
            if channel in self.tools:
                results[channel] = self.tools[channel](state)
        return {"send_results": results}
```

## 5. 레지스트리 — 조립과 provider 결정을 한곳에

환경변수(`SUMMARIZER_PROVIDER`, 활성 채널 목록 등)를 읽어 에이전트를 조립하는 코드를 한 파일에 모은다. `graph/pipeline.py`나 `main.py`는 이 레지스트리만 import해서 쓰면 되고, 어떤 provider/채널이 켜져 있는지는 신경 쓸 필요가 없다.

```python
# agents/registry.py
import os

from tools.news_fetch import fetch_news
from tools.summarize import summarize_claude
from tools.email_send import send_email
# from tools.summarize_hf import summarize_hf   # LoRA 배포 후 추가
# from tools.discord_send import send_discord   # 백로그 #9 구현 후 추가

from .collector import CollectorAgent
from .distributor import DistributorAgent
from .orchestrator import OrchestratorAgent
from .summarizer import SummarizerAgent


def build_agents() -> dict:
    return {
        "orchestrator": OrchestratorAgent(name="orchestrator"),
        "collector": CollectorAgent(name="collector", tools={"fetch_news": fetch_news}),
        "summarizer": SummarizerAgent(
            name="summarizer",
            tools={"summarize_claude": summarize_claude},
            provider=os.getenv("SUMMARIZER_PROVIDER", "anthropic"),
            shadow=os.getenv("SUMMARIZER_SHADOW", "false").lower() == "true",
        ),
        "distributor": DistributorAgent(
            name="distributor",
            tools={"email": send_email},
            channels=os.getenv("DISTRIBUTE_CHANNELS", "email").split(","),
        ),
    }
```

## 6. graph/pipeline.py와의 연결

노드 함수는 이제 tools를 직접 부르지 않고 해당 역할의 에이전트에 위임한다. 그래프 구조(어떤 노드가 어떤 순서로 연결되는지) 자체는 그대로 LangGraph가 담당하고, "다음에 어디로 갈지"의 판단 로직만 `OrchestratorAgent`로 옮긴다.

```python
# graph/pipeline.py (발췌, 변경되는 부분만)
from agents.registry import build_agents

agents = build_agents()

def collector_node(state):
    return agents["collector"].run(state)

def summarizer_node(state):
    return agents["summarizer"].run(state)

def distributor_node(state):
    return agents["distributor"].run(state)

# ...
graph.add_conditional_edges(
    "store_raw",
    agents["orchestrator"].decide_after_store,
    {"summarize": "summarize", "end": END},
)
```

## 7. 이 구조가 해결하는 것

새 provider나 채널을 추가할 때 고칠 파일이 명확해진다. 요약 모델을 바꾸고 싶으면 `summarizer.py`와 `registry.py`만, Discord를 추가하고 싶으면 `distributor.py`와 `tools/discord_send.py`만 건드리면 되고 나머지 에이전트·그래프 구조는 그대로다. 라우팅 판단(오케스트레이터 역할)이 그래프 파일 여기저기 흩어진 람다가 아니라 `OrchestratorAgent` 한 클래스에 모여 있어, 나중에 "이 조건일 땐 이렇게 판단한다"는 규칙이 늘어나도 추적하기 쉽다. 그리고 각 에이전트가 자기 몫의 `tools`만 주입받는 구조라, 코드만 봐도 "수집 에이전트는 발송 tool에 접근할 수 없다"는 게 보장된다.

기존 `tools/*.py`, `db/`, `graph/pipeline.py`의 전체 골격은 바꿀 필요 없고, `agents/` 패키지 하나를 새로 추가하고 `graph/pipeline.py`의 노드 본문만 얇게 리팩터링하면 되는 정도라 마이그레이션 부담도 크지 않다.
