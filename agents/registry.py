"""환경변수를 읽어 에이전트를 조립하는 곳.

graph/pipeline.py나 main.py는 build_agents()만 호출해서 쓰면 되고, 어떤
provider/채널이 켜져 있는지는 신경 쓸 필요가 없다. 새 provider나 채널을
추가할 때 고칠 곳이 이 파일과 해당 에이전트 하나로 명확해진다.
"""
import os

from db.db import log_send, save_digest
from tools.email_send import send_email
from tools.insight import extract_insights
from tools.news_fetch import fetch_news
from tools.summarize import summarize_articles

from .collector import CollectorAgent
from .distributor import DistributorAgent
from .orchestrator import OrchestratorAgent
from .summarizer import SummarizerAgent


def build_agents() -> dict:
    return {
        "orchestrator": OrchestratorAgent(name="orchestrator"),
        "collector": CollectorAgent(
            name="collector",
            tools={"fetch_news": fetch_news},
        ),
        "summarizer": SummarizerAgent(
            name="summarizer",
            tools={
                "summarize": summarize_articles,
                "insight": extract_insights,
                "save_digest": save_digest,
                # "summarize_hf": ...,  # LoRA 배포 후 등록. SUMMARIZER_PROVIDER=hf로 전환.
            },
            provider=os.getenv("SUMMARIZER_PROVIDER", "anthropic"),
        ),
        "distributor": DistributorAgent(
            name="distributor",
            tools={
                "email": send_email,
                "log_send": log_send,
                # "discord": ...,  # 백로그 #9 구현 후 등록
            },
            channels=os.getenv("DISTRIBUTE_CHANNELS", "email").split(","),
        ),
    }
