# ==========================================
# 뉴스레터 생성
# ==========================================

from summarizer import summarize_article


def create_newsletter(category_results):
    """
    카테고리별 검색 결과를 받아
    하나의 뉴스레터 형태로 만든다.
    """

    newsletter = []

    newsletter.append("=" * 70)
    newsletter.append("오늘의 맞춤 뉴스레터")
    newsletter.append("=" * 70)

    newsletter.append("")
    newsletter.append(
        "안녕하세요! 브리피입니다.\n"
        "오늘의 주요 소식을 전해드립니다."
    )


    # ======================================
    # 카테고리별 뉴스 생성
    # ======================================

    for category, articles in category_results.items():

        newsletter.append("")
        newsletter.append("-" * 70)
        newsletter.append(f"[{category}]")
        newsletter.append("-" * 70)


        # 해당 카테고리에 기사가 없는 경우
        if not articles:

            newsletter.append(
                "오늘은 관련 기사가 없습니다."
            )

            continue


        # ==================================
        # 기사별 요약
        # ==================================

        for index, article in enumerate(
            articles,
            start=1
        ):

            title = article[0]
            content = article[1]
            source_url = article[2]


            # 기사 요약
            result = summarize_article(
                title,
                content
            )


            newsletter.append("")
            newsletter.append(
                f"{index}. {result['title']}"
            )

            newsletter.append("")
            newsletter.append(
                result["summary"]
            )

            newsletter.append("")
            newsletter.append(
                f"원문: {source_url}"
            )


    newsletter.append("")
    newsletter.append("=" * 70)
    newsletter.append(
        "오늘의 뉴스가 도움이 되었길 바랍니다."
    )
    newsletter.append("=" * 70)


    return "\n".join(newsletter)