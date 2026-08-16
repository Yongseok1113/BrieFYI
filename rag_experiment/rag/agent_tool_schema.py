"""
RAG 에이전트 Tool 스펙 (JSON 형태로 내보내기)

agent_search.py의 search_news() 함수를
에이전트(LLM)가 이해할 수 있는 "설명서(JSON)" 형태로 만든다.

에이전트는 파이썬 코드를 읽고 이해하는 게 아니라,
"이 도구의 이름은 뭐고, 어떤 상황에 쓰는 거고,
파라미터는 뭐가 있는지"를 JSON 형식으로 미리 정리해둔 걸 보고
"이 도구를 이렇게 호출해야겠다"고 판단한다.

이 JSON 형식은 Anthropic(Claude) API의 tool-use 방식과
OpenAI(GPT) API의 function-calling 방식이 구조가 거의 같아서,
이 파일 하나로 둘 다 커버할 수 있다.
"""

import json

# 스키마(설명서)와 실제 함수가 따로 놀지 않도록
# 같은 파일에서 import해서 연결해둔다.
from rag.agent_search import search_news


# ==========================================
# 1. Tool 스펙 정의
# ==========================================
#
# 이 딕셔너리 하나가 "search_news라는 도구는
# 이렇게 생겼다"를 설명하는 전체 명세서다.

SEARCH_NEWS_TOOL_SCHEMA = {

    # 에이전트가 이 도구를 부를 때 쓸 이름.
    # 실제 파이썬 함수 이름(search_news)이랑 맞춰뒀다.
    "name": "search_news",

    # description: 에이전트가 "이 상황에 이 도구를 써야겠다"고
    # 판단하는 데 가장 중요한 부분.
    # 사람이 읽어도 이해되게, 이 도구가 언제 필요한지 명확히 적는다.
    "description": (
        "대분류/도메인/엔티티/이벤트로 필터링된 "
        "경제·기술 관련 뉴스 기사를 검색한다. "
        "예: '엔비디아 AI 투자 관련 뉴스 찾아줘' 같은 요청에 사용한다. "
        "정치/스포츠/연예 등 비즈니스와 무관한 뉴스는 다루지 않는다."
    ),

    # input_schema: 이 도구를 부를 때 넘겨야 하는
    # 파라미터들의 이름/타입/설명을 정의한다.
    # 이건 JSON Schema라는 정해진 표준 형식을 따른다.
    "input_schema": {

        "type": "object",

        # properties: 실제 파라미터 목록.
        # agent_search.py의 search_news() 함수 파라미터랑
        # 이름과 의미를 정확히 맞춰야 한다.
        "properties": {

            "query": {
                "type": "string",
                "description": "자연어 검색어. 예: '엔비디아 AI 투자 뉴스'",
            },

            "category": {
                "type": "string",
                # enum: "이 값들 중 하나만 골라야 한다"는 제약.
                # classify.py의 CATEGORIES 리스트랑 동일하게 맞춰야
                # 에이전트가 존재하지 않는 카테고리를
                # 잘못 요청하는 걸 막을 수 있다.
                "enum": ["경제", "기술", "금융", "산업", "기타"],
                "description": "대분류 필터. 지정 안 하면 전체 대상 검색.",
            },

            "domain": {
                "type": "string",
                "enum": [
                    "반도체", "AI", "배터리", "바이오", "자동차",
                    "부동산", "증시", "통화정책", "스타트업", "클라우드",
                    "로봇", "에너지", "커머스", "게임", "기타",
                ],
                "description": "도메인 필터. 지정 안 하면 전체 대상 검색.",
            },

            "entity": {
                "type": "string",
                # entity는 회사/인물명이 계속 늘어날 수 있어서
                # category/domain처럼 enum(고정 목록)으로 막지 않고
                # 자유 문자열로 둔다.
                "description": "특정 기업/기관/인물명 필터. 예: 'NVIDIA', '삼성전자'",
            },

            "event": {
                "type": "string",
                "enum": [
                    "투자", "인수합병", "실적", "규제", "출시",
                    "제휴", "구조조정", "상장", "특허", "기타",
                ],
                "description": "이벤트 유형 필터. 지정 안 하면 전체 대상 검색.",
            },

            "top_k": {
                "type": "integer",
                "description": "반환할 최대 결과 개수 (기본값 5)",
                "default": 5,
            },
        },

        # required: 위 파라미터 중에서 "반드시 있어야 하는 것"만 적는다.
        # query만 필수고, 나머지 필터는 전부 선택사항이라
        # 리스트에 query만 들어간다.
        "required": ["query"],
    },
}


# ==========================================
# 2. 실제 도구 실행 함수 연결
# ==========================================

def execute_search_news(arguments: dict) -> list:
    """
    에이전트가 SEARCH_NEWS_TOOL_SCHEMA를 보고
    "이 도구를 이 파라미터로 호출하고 싶다"고 결정하면,
    실제로 호출되는 함수.

    arguments:
        에이전트가 넘겨준 파라미터 딕셔너리.
        예: {"query": "엔비디아 투자 뉴스", "domain": "AI"}

    Returns:
        search_news() 함수의 결과 (JSON 직렬화 가능한 리스트)
    """

    # arguments.get("파라미터명", 기본값)
    #   → 에이전트가 그 파라미터를 안 넘겼을 수도 있으니
    #     안전하게 기본값과 함께 꺼낸다.
    return search_news(
        query=arguments.get("query", ""),
        category=arguments.get("category"),
        domain=arguments.get("domain"),
        entity=arguments.get("entity"),
        event=arguments.get("event"),
        top_k=arguments.get("top_k", 5),
    )


# ==========================================
# 3. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("Tool Schema 확인 (에이전트에게 보여줄 JSON)")
    print("=" * 60)

    # 실제로 JSON 문자열로 잘 변환되는지 확인.
    # 이 출력물을 그대로 복사해서
    # Claude/GPT API의 tools 파라미터에 넣으면 된다.
    print(
        json.dumps(
            SEARCH_NEWS_TOOL_SCHEMA,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\n" + "=" * 60)
    print("execute_search_news() 실제 호출 테스트")
    print("=" * 60)

    # 에이전트가 이런 식으로 파라미터를 넘겨줬다고 가정하고 테스트.
    fake_agent_call = {
        "query": "김민석 관련 뉴스",
        "top_k": 2,
    }

    results = execute_search_news(fake_agent_call)

    print(f"\n결과: {len(results)}개")

    for result in results:
        print(f"- {result['title']} (점수: {result['relevance_score']})")

    print("\n" + "=" * 60)
    print("Tool Schema 테스트 완료")
    print("=" * 60)