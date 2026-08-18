"""채널별 발송을 담당하는 에이전트.

email 채널은 config.EMAIL_RECIPIENTS(EMAIL_TO를 쉼표로 나눈 목록)의 각 주소에 개별
발송하고, 성공/실패를 주소별로 log_send에 남긴다 — 구 tools/deployment.py가 하던
다중 수신자 배포 로직을 여기로 흡수했다(수신자 소스만 로컬 JSON 파일 대신
config.py의 기존 env 관례를 따르도록 바꿨다). Discord 발송(백로그 #9)을 추가할 때도
오케스트레이터나 다른 에이전트는 건드릴 필요 없이, registry.py의 tools에 "discord"를
등록하고 channels 목록에 이름만 추가하면 된다.
"""
from config import config

from .base import Agent


class DistributorAgent(Agent):
    def __init__(self, name: str, tools: dict, channels: list[str]):
        super().__init__(name=name, tools=tools)
        self.channels = channels

    def run(self, state: dict) -> dict:
        subject = f"[{state['digest_date']}] {state['keyword']} 뉴스·기술 다이제스트"

        if "email" not in self.channels or "email" not in self.tools:
            return {}

        recipients = config.EMAIL_RECIPIENTS or ([config.EMAIL_TO] if config.EMAIL_TO else [])
        results = []
        for recipient in recipients:
            try:
                result = self.tools["email"](subject, state["email_html"], to=recipient)
                self.tools["log_send"](state["digest_id"], "email", recipient, "success")
                results.append({"recipient": recipient, "status": "success", "result": result})
            except Exception as exc:  # noqa: BLE001 - MVP 단계는 넓게 잡고 로그만 남긴다
                self.tools["log_send"](state["digest_id"], "email", recipient, "failed", str(exc))
                results.append({"recipient": recipient, "status": "failed", "error": str(exc)})

        success_count = sum(r["status"] == "success" for r in results)
        output = {
            "send_result": {
                "recipients": results,
                "success_count": success_count,
                "total_count": len(results),
            }
        }
        if results and success_count == 0:
            # main.py의 run_digest()가 result["error"]를 보고 종료 코드를 결정하므로,
            # 전원 발송 실패를 error로도 알려야 daily-digest가 조용히 "성공"으로 끝나지 않는다.
            output["error"] = f"이메일 발송 전원 실패 ({len(results)}명)"
        return output
