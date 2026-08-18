from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import psycopg2


# ==========================================
# 1. 임베딩 모델
# ==========================================

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ==========================================
# 2. 리랭커 모델
# ==========================================

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=False
)


# ==========================================
# 3. 관심분야 목록
# ==========================================

categories = {
    "1": "정치",
    "2": "경제",
    "3": "사회",
    "4": "국제",
    "5": "스포츠",
    "6": "연예",
    "7": "문화",
    "8": "과학/IT"
}


# ==========================================
# 4. 관심분야 선택
# ==========================================

print("\n" + "=" * 70)
print("관심분야를 선택하세요.")
print("여러 개를 선택할 수 있습니다.")
print("=" * 70)

for number, category in categories.items():
    print(f"{number}. {category}")

print("\n예시: 1,5,8")
print("전체 선택: all")

selected_input = input("\n선택: ").strip()


# ==========================================
# 5. 선택한 카테고리 처리
# ==========================================

if selected_input.lower() == "all":

    selected_categories = list(categories.values())

else:

    selected_numbers = [
        number.strip()
        for number in selected_input.split(",")
    ]

    selected_categories = [
        categories[number]
        for number in selected_numbers
        if number in categories
    ]


# 잘못된 입력 처리
if not selected_categories:

    print("\n올바른 관심분야를 선택해주세요.")
    exit()


print("\n선택한 관심분야:")

for category in selected_categories:
    print(f"- {category}")


# ==========================================
# 6. PostgreSQL 연결
# ==========================================

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_test",
    user="rag_user",
    password="rag_password"
)

cur = conn.cursor()


# ==========================================
# 7. 카테고리별 결과 저장 공간
# ==========================================

category_results = {}


# ==========================================
# 8. 선택한 카테고리별 검색
# ==========================================

for category in selected_categories:

    print("\n" + "=" * 70)
    print(f"관심분야: {category}")
    print("=" * 70)


    # --------------------------------------
    # 카테고리 이름을 검색 질문으로 사용
    # --------------------------------------

    category_query = f"{category} 관련 최신 뉴스"


    # --------------------------------------
    # 카테고리 검색어 임베딩
    # --------------------------------------

    query_embedding = embedding_model.encode(
        category_query
    ).tolist()


    # --------------------------------------
    # pgvector 1차 검색
    # --------------------------------------
    #
    # 관련성이 높은 후보를 넓게 가져온다.
    #

    cur.execute(
        """
        SELECT
            title,
            content,
            source_url,
            embedding <=> %s::vector AS distance
        FROM rag_documents
        ORDER BY embedding <=> %s::vector
        LIMIT 15
        """,
        (
            query_embedding,
            query_embedding
        )
    )


    results = cur.fetchall()


    # 후보가 없는 경우
    if not results:

        category_results[category] = []

        print("후보 기사가 없습니다.")

        continue


    # ======================================
    # 9. BGE Reranker 입력 생성
    # ======================================

    rerank_pairs = [
        [category_query, result[1]]
        for result in results
    ]


    # ======================================
    # 10. BGE Reranker 실행
    # ======================================

    rerank_scores = reranker.compute_score(
        rerank_pairs,
        normalize=True
    )


    # 결과가 하나일 경우 숫자로 반환될 수 있음
    if isinstance(rerank_scores, float):

        rerank_scores = [rerank_scores]


    # ======================================
    # 11. 검색 결과 + 리랭커 점수 결합
    # ======================================

    reranked_results = []


    for result, score in zip(
        results,
        rerank_scores
    ):

        title = result[0]
        content = result[1]
        source_url = result[2]
        distance = result[3]


        reranked_results.append(
            (
                title,
                content,
                source_url,
                distance,
                score
            )
        )


    # ======================================
    # 12. 리랭커 점수가 높은 순서로 정렬
    # ======================================

    reranked_results.sort(
        key=lambda x: x[4],
        reverse=True
    )


    # ======================================
    # 13. 같은 기사 중복 제거
    # ======================================

    unique_results = []

    seen_urls = set()


    for result in reranked_results:

        source_url = result[2]


        if source_url in seen_urls:
            continue


        seen_urls.add(source_url)

        unique_results.append(result)


        # 카테고리별 최대 3개
        if len(unique_results) >= 3:
            break


    # 카테고리별 결과 저장
    category_results[category] = unique_results


# ==========================================
# 14. 최종 뉴스레터 후보 출력
# ==========================================

print("\n\n" + "=" * 70)
print("맞춤 뉴스레터 후보 기사")
print("=" * 70)


has_results = False


for category in selected_categories:

    print("\n" + "=" * 70)
    print(f"[{category}]")
    print("=" * 70)


    results = category_results.get(
        category,
        []
    )


    # 해당 카테고리에 결과가 없는 경우
    if not results:

        print("관련 기사를 찾지 못했습니다.")

        continue


    has_results = True


    # --------------------------------------
    # 카테고리별 기사 출력
    # --------------------------------------

    for index, result in enumerate(
        results,
        start=1
    ):

        title = result[0]
        content = result[1]
        source_url = result[2]
        distance = result[3]
        rerank_score = result[4]


        print("\n" + "-" * 70)

        print(f"[기사 {index}]")

        print(f"리랭커 점수: {rerank_score:.4f}")

        print(f"기사 제목: {title}")

        print("\n기사 내용:")

        print(content)

        print("\n원문:")

        print(source_url)


# ==========================================
# 15. 전체 결과가 없는 경우
# ==========================================

if not has_results:

    print("\n선택한 관심분야에서 관련 기사를 찾지 못했습니다.")


# ==========================================
# 16. DB 연결 종료
# ==========================================

cur.close()
conn.close()


print("\n" + "=" * 70)
print("맞춤 뉴스 검색 완료")
print("=" * 70)