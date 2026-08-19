"""
BrieFYI 구독 신청 API (실 서비스 버전)

brief-signup.html이 이 서버로 실제 신청 데이터를 보낸다.

실행 (반드시 프로젝트 최상단 BrieFYI 폴더에서):
    uvicorn signup_service.app:app --reload --port 8000
"""

from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from config import config
from db.db import init_db as init_team_schema

from . import subscribers as subscriber_db


# rag_experiment/rag/classify.py의 현재 taxonomy(경제/기술/금융/산업)에는
# 스포츠/정치가 없다. demo 브랜치 화면 데모용으로 넣어둔 것이고,
# develop 병합 전에 classify.py taxonomy를 같이 넓힐지 팀과 논의 필요.
VALID_CATEGORIES = {"경제", "기술", "금융", "산업", "스포츠", "정치"}

RESEND_URL = "https://api.resend.com/emails"
SITE_URL = "https://briefyi.netlify.app/"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_team_schema()
    yield


app = FastAPI(title="BrieFYI Signup API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)


def send_welcome_email(to_email: str, categories: list[str]) -> None:
    """
    신규 구독자에게 환영 메일을 보낸다.
    """

    if not config.RESEND_API_KEY or not config.EMAIL_FROM:
        print("RESEND_API_KEY 또는 EMAIL_FROM 미설정 - 환영 메일 발송을 건너뜀")
        return

    categories_text = ", ".join(categories)

    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {config.RESEND_API_KEY}"},
            json={
                "from": config.EMAIL_FROM,
                "to": [config.EMAIL_TO],
                "subject": "BrieFYI 구독을 환영합니다! 🎉 첫 소식을 기다려주세요.",
                "html": (
                    "<p>안녕하세요, BrieFYI 구독자님!</p>"
                    "<p>BrieFYI 뉴스레터를 구독해 주셔서 진심으로 감사합니다."
                    "앞으로 BrieFYI에서는 매일 아침마다 관심 분야에 대한"
                    " 알차고 유익한 뉴스를 전해드릴게요.</p>"
                    "<p>📌 앞으로 이런 내용을 보내드려요!</p>"
                    f"<p>관심분야 : <b>{categories_text}</b></p>"
                    f'<p>👉 <a href="{SITE_URL}">{SITE_URL}</a></p>'
                    "<p>혹시 메일이 보이지 않고 스팸함에 들어있다면,"
                    f" {config.EMAIL_FROM}을 주소록에 추가해 주세요!</p>"
                    "<p>감사합니다.<br>BrieFYI 드림</p>"
                ),
            },
            timeout=15,
        )

        if not resp.ok:
            print(f"환영 메일 발송 실패 ({resp.status_code}): {resp.text}")

    except Exception as e:
        print(f"환영 메일 발송 중 오류: {e}")


class SubscribeRequest(BaseModel):
    email: EmailStr
    categories: list[str] = []
    hp: str = ""


@app.post("/subscribe")
def subscribe(payload: SubscribeRequest):

    if payload.hp:
        return {"status": "ok"}

    categories = [c for c in payload.categories if c in VALID_CATEGORIES]

    if not categories:
        raise HTTPException(
            status_code=400,
            detail="관심 분야를 하나 이상 선택해야 합니다.",
        )

    result = subscriber_db.upsert_subscriber(payload.email, categories)

    if result["created"]:
        send_welcome_email(payload.email, categories)

    return {
        "status": "ok",
        "created": result["created"],
        "categories": categories,
    }


class UnsubscribeRequest(BaseModel):
    email: EmailStr


@app.delete("/subscribe")
def unsubscribe(payload: UnsubscribeRequest):
    deleted = subscriber_db.delete_subscriber(payload.email)
    return {"status": "ok", "deleted": deleted}


@app.get("/health")
def health():
    return {"status": "ok", "subscribers": subscriber_db.get_subscriber_count()}