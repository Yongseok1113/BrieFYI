"""
RAG 메타데이터 분류 (인덱스 구조 개선)

기사 제목 + 본문을 Claude(Anthropic API)한테 보내서
"이 기사는 대분류/도메인/엔티티/이벤트가 뭐야?"라고 물어보고
답변을 딕셔너리(dict) 형태로 돌려받는다.

검색할 때 "AI 뉴스만 보여줘" 같은 필터링을 하려면
기사마다 이 분류 정보가 미리 붙어있어야 한다.

기사를 DB에 저장하기 "직전"에, 기사 1개당 딱 1번만 호출된다.
"""

import json
import os

# --------------------------------------------------
# .env 파일 불러오기
#
# BrieFYI 프로젝트의 .env 파일은 최상단(BrieFYI/.env)에 있다.
# 현재 작업 파일은 하위 폴더 안에 있어서 .env 파일 존재를 모름
#
# find_dotenv()
#   → 지금 있는 폴더부터 시작해서 상위 폴더로 계속 올라가면서
#     .env 파일을 자동으로 찾아 그 경로를 알려준다.
# load_dotenv(그 경로)
#   → 찾은 .env 파일 안의 내용(ANTHROPIC_API_KEY=... 등)을
#     전부 읽어서, 파이썬이 os.environ.get(...)으로
#     꺼내 쓸 수 있게 등록해준다.
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Anthropic: Claude를 API로 호출하기 위한 공식 파이썬 라이브러리.
from anthropic import Anthropic


# ==========================================
# 1. Claude API 클라이언트 설정
# ==========================================

# 위에서 load_dotenv()를 실행했기 때문에
# os.environ.get("ANTHROPIC_API_KEY")가 .env에 적어둔
# 실제 키 값을 제대로 가져올 수 있다.
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# ==========================================
# 2. 고정 분류 목록 (taxonomy)
# ==========================================
# 미리 정해둔 리스트에 있는 것 중에서만 고르도록 Claude에게 시킴

CATEGORIES = ["경제", "기술", "금융", "산업", "기타"]

DOMAINS = [
    "반도체", "AI", "배터리", "바이오", "자동차",
    "부동산", "증시", "통화정책", "스타트업", "클라우드",
    "로봇", "에너지", "커머스", "게임", "기타",
]

EVENTS = [
    "투자", "인수합병", "실적", "규제", "출시",
    "제휴", "구조조정", "상장", "특허", "기타",
]


# ==========================================
# 3. Claude에게 보낼 프롬프트(질문) 양식
# ==========================================

CLASSIFY_PROMPT = """다음 뉴스 기사를 분류해줘.

제목: {title}
본문: {content}

아래 JSON 형식으로만 답해. 다른 설명은 하지 마.

{{
  "category": "다음 중 하나: {categories}",
  "domain": "다음 중 하나: {domains}",
  "entities": ["기사에 등장하는 핵심 기업/기관/인물명, 최대 5개"],
  "event": "다음 중 하나: {events}"
}}
"""


# ==========================================
# 4. 실제 분류 함수
# ==========================================

def classify_article(title: str, content: str) -> dict:
    """
    기사 제목(title)과 본문(content)을 받아서
    Claude에게 분류를 요청하고, 결과를 딕셔너리로 돌려준다.
    """

    prompt = CLASSIFY_PROMPT.format(
        title=title,
        content=content[:1500],
        categories=", ".join(CATEGORIES),
        domains=", ".join(DOMAINS),
        events=", ".join(EVENTS),
    )

    try:

        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        return {
            "category": result.get("category") or "기타",
            "domain": result.get("domain") or "기타",
            "entities": result.get("entities") or [],
            "event": result.get("event") or "기타",
        }

    except Exception as e:

        print(f"분류 실패 (기본값 사용): {e}")

        return {
            "category": "기타",
            "domain": "기타",
            "entities": [],
            "event": "기타",
        }


# ==========================================
# 5. 이 파일을 직접 실행했을 때만 동작하는 테스트 코드
# ==========================================

if __name__ == "__main__":

    result = classify_article(
        title="엔비디아, AI 반도체 스타트업에 대규모 투자",
        content=(
            "엔비디아가 인공지능 반도체 스타트업에 "
            "수억 달러 규모의 투자를 단행했다고 발표했다."
        ),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))