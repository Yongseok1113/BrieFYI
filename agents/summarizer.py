"""요약과 인사이트/비즈니스 시사점 추출을 함께 담당하는 에이전트.

run_summarize가 원래 summarize_node, run_insight가 원래 insight_node의 역할을
그대로 옮겨온 것이라 동작은 바뀌지 않는다. provider는 지금은 "anthropic"
하나뿐이지만, 자체 LoRA 모델을 Hugging Face에 배포하면 이 필드로 Claude와
새 엔드포인트를 스위치하도록 설계했다(lora-finetune-summarization-design.md
5절의 SUMMARIZER_PROVIDER 참고). 실제로 hf provider를 쓰려면 registry.py에서
tools에 "summarize_hf"를 등록하고 아래 run_summarize에서 분기만 추가하면 된다.
"""
from .base import Agent


class SummarizerAgent(Agent):
    def __init__(self, name: str, tools: dict, provider: str = "anthropic"):
        super().__init__(name=name, tools=tools)
        self.provider = provider

    def run_summarize(self, state: dict) -> dict:
        summarize_fn = self.tools["summarize_hf"] if self.provider == "hf" and "summarize_hf" in self.tools else self.tools["summarize"]
        summaries = summarize_fn(state["raw_articles"])
        return {"summaries": summaries}

    def run_insight(self, state: dict) -> dict:
        insight = self.tools["insight"](state["summaries"])
        digest_id = self.tools["save_digest"](
            state["digest_date"], state["keyword"], state["summaries"], insight
        )
        return {"insight": insight, "digest_id": digest_id}
