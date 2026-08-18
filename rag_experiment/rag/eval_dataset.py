"""
RAG 검색 성능 평가 - 데이터셋

"정답을 미리 아는" 가짜 기사들을 만들어서 DB에 넣고,
"이 질문의 정답은 이 기사다"라는 테스트 세트를 함께 정의한다.

가짜 기사를 쓰는 이유
지금 DB에 있는 실제 기사는 전부 정치 뉴스라서 category가 다 "기타"다.
그러면 category/domain/event 필터가 진짜 잘 작동하는지 확인할 방법이 없다.
그래서 대분류/도메인/이벤트가 골고루 섞인 가짜 기사를 만들어서,
"필터를 걸었을 때 정말 그 필터에 맞는 기사만 나오는지"를
명확하게 확인할 수 있게 만든 것이다.

나중에 진짜 경제/기술 기사가 많이 쌓이면,
이 파일의 EVAL_QUERIES 부분만 진짜 기사 기준으로 바꿔서
그대로 재사용하면 된다.
"""

from rag.indexer import index_chunks
from rag.db_config import DB_CONFIG

import psycopg2


# ==========================================
# 1. 평가용 가짜 기사 12개
# ==========================================
# 대분류(경제/기술/금융/산업)와 도메인, 이벤트가
# 최대한 겹치지 않게 골고루 섞어서 만들었다.
#
# source_url을 전부 "https://example.com/eval/"로 시작하게
# 만들어둔 이유: 나중에 이 테스트 데이터만 골라서
# 한 번에 지우기 쉽게 하기 위함 (구분자 역할).

EVAL_ARTICLES = [
    {
        "title": "엔비디아, AI 반도체 스타트업에 대규모 투자",
        "content": "엔비디아가 인공지능 반도체 스타트업에 수억 달러 규모의 투자를 단행했다고 발표했다. 이번 투자로 AI 반도체 생태계가 더욱 확장될 전망이다.",
        "source_url": "https://example.com/eval/1",
        "category": "기술", "domain": "AI", "entities": ["엔비디아"], "event": "투자",
    },
    {
        "title": "삼성전자, 3분기 반도체 부문 실적 발표",
        "content": "삼성전자가 3분기 반도체 부문에서 시장 예상치를 웃도는 실적을 기록했다고 밝혔다. 메모리 반도체 가격 회복이 실적 개선의 주요 요인으로 꼽힌다.",
        "source_url": "https://example.com/eval/2",
        "category": "기술", "domain": "반도체", "entities": ["삼성전자"], "event": "실적",
    },
    {
        "title": "OpenAI, AI 스타트업 인수합병 추진",
        "content": "OpenAI가 소규모 AI 스타트업을 인수하는 방안을 추진 중이라고 알려졌다. 이번 인수합병을 통해 자체 기술 역량을 강화할 계획이다.",
        "source_url": "https://example.com/eval/3",
        "category": "기술", "domain": "스타트업", "entities": ["OpenAI"], "event": "인수합병",
    },
    {
        "title": "테슬라 배터리 공장, 환경 규제 강화 직면",
        "content": "테슬라의 배터리 생산 공장이 새로운 환경 규제 적용을 앞두고 있다. 정부는 배터리 제조 과정의 오염물질 배출 기준을 강화할 방침이다.",
        "source_url": "https://example.com/eval/4",
        "category": "산업", "domain": "배터리", "entities": ["테슬라"], "event": "규제",
    },
    {
        "title": "카카오, 이커머스 플랫폼과 제휴 발표",
        "content": "카카오가 대형 이커머스 플랫폼과 전략적 제휴를 맺었다고 발표했다. 이번 제휴로 카카오의 커머스 사업이 한층 확대될 전망이다.",
        "source_url": "https://example.com/eval/5",
        "category": "산업", "domain": "커머스", "entities": ["카카오"], "event": "제휴",
    },
    {
        "title": "한국은행, 기준금리 동결 결정",
        "content": "한국은행이 이번 금융통화위원회에서 기준금리를 동결하기로 결정했다. 물가 안정과 경기 상황을 종합적으로 고려한 결정이라고 설명했다.",
        "source_url": "https://example.com/eval/6",
        "category": "금융", "domain": "통화정책", "entities": ["한국은행"], "event": "기타",
    },
    {
        "title": "쿠팡, 상장 절차 본격 추진",
        "content": "쿠팡이 상장을 위한 절차를 본격적으로 추진하고 있다고 밝혔다. 투자자들의 관심이 높아지면서 관련 업계도 주목하고 있다.",
        "source_url": "https://example.com/eval/7",
        "category": "산업", "domain": "커머스", "entities": ["쿠팡"], "event": "상장",
    },
    {
        "title": "네이버 클라우드, AI 기술 특허 출원",
        "content": "네이버 클라우드가 자체 개발한 AI 기술에 대한 특허를 출원했다고 발표했다. 이번 특허는 대규모 언어 모델 처리 속도를 개선하는 기술이다.",
        "source_url": "https://example.com/eval/8",
        "category": "기술", "domain": "클라우드", "entities": ["네이버"], "event": "특허",
    },
    {
        "title": "현대자동차, 신형 물류 로봇 출시",
        "content": "현대자동차가 공장 자동화를 위한 신형 물류 로봇을 출시했다고 밝혔다. 이 로봇은 생산 효율을 크게 높일 것으로 기대된다.",
        "source_url": "https://example.com/eval/9",
        "category": "산업", "domain": "로봇", "entities": ["현대자동차"], "event": "출시",
    },
    {
        "title": "SK하이닉스, 조직 개편 및 구조조정 발표",
        "content": "SK하이닉스가 사업 효율화를 위한 조직 개편과 일부 부문 구조조정을 발표했다. 반도체 시장 변화에 대응하기 위한 조치로 풀이된다.",
        "source_url": "https://example.com/eval/10",
        "category": "기술", "domain": "반도체", "entities": ["SK하이닉스"], "event": "구조조정",
    },
    {
        "title": "카카오뱅크, 핀테크 스타트업에 투자",
        "content": "카카오뱅크가 유망 핀테크 스타트업에 지분 투자를 단행했다고 밝혔다. 디지털 금융 서비스 경쟁력 강화를 위한 행보로 분석된다.",
        "source_url": "https://example.com/eval/11",
        "category": "금융", "domain": "스타트업", "entities": ["카카오뱅크"], "event": "투자",
    },
    {
        "title": "롯데, 부동산 개발사 인수합병 완료",
        "content": "롯데그룹이 중견 부동산 개발사에 대한 인수합병을 완료했다고 발표했다. 이번 인수를 통해 부동산 개발 사업 포트폴리오를 확장한다.",
        "source_url": "https://example.com/eval/12",
        "category": "산업", "domain": "부동산", "entities": ["롯데"], "event": "인수합병",
    },
]


# ==========================================
# 2. 평가용 테스트 질문 10개
# ==========================================
#   query: 실제로 검색해볼 자연어 문장
#   expected_title: 이 질문의 "정답" 기사 제목
#                    (retrieve_documents 결과 안에 이 제목이 있어야 성공)
#   filters: 이 질문을 검색할 때 같이 걸어줄 필터.
#            없으면 빈 딕셔너리 {} (필터 없이 순수 벡터 검색만)

EVAL_QUERIES = [
    {
        "query": "엔비디아 AI 투자 뉴스",
        "expected_title": "엔비디아, AI 반도체 스타트업에 대규모 투자",
        "filters": {},
    },
    {
        "query": "삼성전자 실적 발표",
        "expected_title": "삼성전자, 3분기 반도체 부문 실적 발표",
        "filters": {},
    },
    {
        "query": "AI 스타트업 인수 소식",
        "expected_title": "OpenAI, AI 스타트업 인수합병 추진",
        # domain 필터를 같이 걸어서 테스트.
        # 이 필터가 없어도 벡터 검색만으로 찾을 수 있는지,
        # 필터가 있으면 더 정확해지는지 둘 다 비교해볼 수 있다.
        "filters": {"domain": "스타트업"},
    },
    {
        "query": "배터리 공장 환경 규제",
        "expected_title": "테슬라 배터리 공장, 환경 규제 강화 직면",
        "filters": {"category": "산업"},
    },
    {
        "query": "카카오 이커머스 제휴",
        "expected_title": "카카오, 이커머스 플랫폼과 제휴 발표",
        "filters": {},
    },
    {
        "query": "한국은행 기준금리 소식",
        "expected_title": "한국은행, 기준금리 동결 결정",
        "filters": {},
    },
    {
        "query": "쿠팡 상장 준비",
        "expected_title": "쿠팡, 상장 절차 본격 추진",
        "filters": {"event": "상장"},
    },
    {
        "query": "네이버 클라우드 특허",
        "expected_title": "네이버 클라우드, AI 기술 특허 출원",
        "filters": {},
    },
    {
        "query": "현대차 로봇 신제품",
        "expected_title": "현대자동차, 신형 물류 로봇 출시",
        "filters": {"domain": "로봇"},
    },
    {
        "query": "SK하이닉스 인력 구조조정",
        "expected_title": "SK하이닉스, 조직 개편 및 구조조정 발표",
        "filters": {},
    },
]


# ==========================================
# 3. 평가 데이터 DB에 넣기
# ==========================================

def seed_eval_data():
    """
    EVAL_ARTICLES 12개를 실제로 DB에 저장한다.

    classify.py(LLM 분류)를 거치지 않고
    category/domain/entities/event 값을 "직접" 넣는다.
    → 정답을 우리가 이미 확정해뒀기 때문에,
      LLM이 잘못 분류할 가능성까지 평가에 섞이지 않게 하기 위함.
      (지금 평가하려는 건 "LLM 분류 정확도"가 아니라
       "검색/필터가 정확히 동작하는가"이기 때문)
    """

    print("평가용 가짜 기사 저장 중...")

    for article in EVAL_ARTICLES:

        index_chunks(
            title=article["title"],
            source_url=article["source_url"],
            # content가 짧아서 chunk 1개로만 저장된다.
            chunks=[article["content"]],
            category=article["category"],
            domain=article["domain"],
            entities=article["entities"],
            event=article["event"],
        )

    print(f"{len(EVAL_ARTICLES)}개 평가용 기사 저장 완료")


# ==========================================
# 4. 평가 데이터 DB에서 지우기
# ==========================================

def cleanup_eval_data():
    """
    평가가 끝난 뒤, 테스트용 가짜 기사를 DB에서 전부 지운다.

    source_url이 "https://example.com/eval/"로 시작하는 것만
    골라서 지우기 때문에, 실제 SBS 기사는 전혀 건드리지 않는다.
    """

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            DELETE FROM rag_documents
            WHERE source_url LIKE 'https://example.com/eval/%'
            """
        )

        deleted_count = cur.rowcount

        conn.commit()

        print(f"평가용 가짜 기사 {deleted_count}개 삭제 완료")

    finally:
        cur.close()
        conn.close()


# ==========================================
# 5. 테스트
# ==========================================
# 이 파일을 직접 실행하면 평가용 데이터를 DB에 심는다.

if __name__ == "__main__":

    print("=" * 60)
    print("평가 데이터셋 준비")
    print("=" * 60)

    seed_eval_data()

    print("\n다음 명령으로 검색 성능 평가를 실행하세요:")
    print("  python -m rag.eval_retriever")
    print("\n평가 끝난 뒤 정리하려면:")
    print("  python -c \"from rag.eval_dataset import cleanup_eval_data; cleanup_eval_data()\"")