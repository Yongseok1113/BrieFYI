"""채널별 발송을 담당하는 에이전트.

지금은 email 채널 하나뿐이라 동작은 원래 send_email_node와 똑같다. Discord
발송(백로그 #9)을 추가할 때도 오케스트레이터나 다른 에이전트는 건드릴 필요
없이, registry.py의 tools에 "discord"를 등록하고 channels 목록에 이름만
추가하면 된다.
"""
from config import config

from .base import Agent


class DistributorAgent(Agent):
    def __init__(self, name: str, tools: dict, channels: list[str]):
        super().__init__(name=name, tools=tools)
        self.channels = channels

    def run(self, state: dict) -> dict:
        subject = f"[{state['digest_date']}] {state['keyword']} 뉴스·기술 다이제스트"

        if "email" in self.channels and "email" in self.tools:
            try:
                result = self.tools["email"](subject, state["email_html"])
                self.tools["log_send"](state["digest_id"], "email", config.EMAIL_TO, "success")
                return {"send_result": result}
            except Exception as exc:  # noqa: BLE001 - MVP 단계는 넓게 잡고 로그만 남긴다
                self.tools["log_send"](state["digest_id"], "email", config.EMAIL_TO, "failed", str(exc))
                return {"error": str(exc)}

        return {}
