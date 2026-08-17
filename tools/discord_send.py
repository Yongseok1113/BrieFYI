"""Discord Webhook 발송 도구 (백로그 #9).

Webhook URL로 POST 한 번이면 끝이라 봇 토큰이나 상시 연결(gateway)이 필요 없다.
다만 이메일이 Jinja2로 HTML을 만드는 것과 달리 Discord는 embed(JSON) 구조를 쓰므로,
같은 인사이트 JSON을 embed로 옮기는 format_discord를 이 모듈에 함께 뒀다.
포맷이 커지면 email_format.py처럼 tools/discord_format.py로 떼어내면 된다.

주의: DISCORD_WEBHOOK_URL의 마지막 경로 조각은 토큰(=비밀)이다. 로그·DB·예외 메시지
어디에도 URL 원문을 남기지 않는다. 발송 이력에는 webhook_label()이 만든 ID만 기록한다.
"""
import requests

from config import config

# Discord embed 제한. 넘기면 400을 돌려주므로 보내기 전에 잘라낸다.
# https://discord.com/developers/docs/resources/message#embed-object-embed-limits
TITLE_LIMIT = 256
DESCRIPTION_LIMIT = 4096
FIELD_NAME_LIMIT = 256
FIELD_VALUE_LIMIT = 1024
FIELD_COUNT_LIMIT = 25
TOTAL_LIMIT = 6000  # embed 하나의 title+description+field 전체 글자 수 합

MAX_INSIGHT_FIELDS = 10
MAX_SUMMARY_FIELDS = 10
EMBED_COLOR = 0x5865F2  # Discord blurple


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _embed_length(embed: dict) -> int:
    return (
        len(embed["title"])
        + len(embed["description"])
        + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
    )


def format_discord(subject: str, summaries: list[dict], insight: dict) -> dict:
    """이메일과 같은 데이터(summaries + insight)를 Discord webhook 페이로드로 바꾼다.

    인사이트를 위에, 주제별 요약을 아래에 붙인 embed 1개로 만든다.
    """
    fields: list[dict] = []

    for i, item in enumerate(insight.get("insights", [])[:MAX_INSIGHT_FIELDS], start=1):
        value = _truncate(item.get("text", ""), FIELD_VALUE_LIMIT - 200)
        source_url = item.get("source_url")
        if source_url:
            value = f"{value}\n[출처]({source_url})"
        fields.append({"name": f"인사이트 {i}", "value": _truncate(value, FIELD_VALUE_LIMIT), "inline": False})

    for s in summaries[:MAX_SUMMARY_FIELDS]:
        fields.append(
            {
                "name": _truncate(f"📰 {s.get('topic_title', '')}", FIELD_NAME_LIMIT),
                "value": _truncate(s.get("summary", ""), FIELD_VALUE_LIMIT),
                "inline": False,
            }
        )

    embed = {
        "title": _truncate(subject, TITLE_LIMIT),
        "description": _truncate(insight.get("business_implication", ""), DESCRIPTION_LIMIT),
        "color": EMBED_COLOR,
        "fields": fields[:FIELD_COUNT_LIMIT],
    }
    # 개별 필드가 다 제한을 지켜도 총합 6000자를 넘으면 거부당한다. 뒤(요약)부터 덜어낸다.
    while embed["fields"] and _embed_length(embed) > TOTAL_LIMIT:
        embed["fields"].pop()

    return {"embeds": [embed]}


def send_discord(subject: str, summaries: list[dict], insight: dict) -> dict:
    """Webhook으로 다이제스트 1건을 발송하고 Discord가 돌려준 메시지 정보를 반환한다.

    `wait=true`를 붙이면 204 대신 200 + 메시지 JSON(id 포함)을 받는다. 이 id가 있어야
    나중에 "무엇이 실제로 올라갔는지"를 확인하거나 메시지를 지울 수 있다.
    """
    if not config.DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL이 설정되지 않았습니다 (.env 확인)")

    resp = requests.post(
        config.DISCORD_WEBHOOK_URL,
        params={"wait": "true"},
        json=format_discord(subject, summaries, insight),
        timeout=15,
    )
    if not resp.ok:
        # 실패 사유(길이 초과, 웹훅 삭제됨 등)는 본문에 담겨 오므로 그대로 노출한다.
        # URL은 토큰을 포함하므로 절대 메시지에 넣지 않는다.
        raise RuntimeError(f"Discord 발송 실패 ({resp.status_code}): {resp.text}")
    return resp.json()


def webhook_label(url: str) -> str:
    """send_log.recipient에 남길 식별자.

    URL은 `.../webhooks/<id>/<token>` 형태고 token은 그 자체로 발송 권한이다.
    이력에는 id만 남겨 어느 웹훅으로 보냈는지 구분만 되게 한다.
    """
    parts = (url or "").rstrip("/").split("/")
    webhook_id = parts[-2] if len(parts) >= 2 and parts[-2] else "unknown"
    return f"discord:webhook/{webhook_id}"
