import requests
from bs4 import BeautifulSoup
import feedparser


# ==========================================
# 1. SBS RSS 주소
# ==========================================

RSS_URL = "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER"


# ==========================================
# 2. SBS RSS에서 기사 목록 가져오기
# ==========================================

def fetch_sbs_rss(max_results=5):

    feed = feedparser.parse(RSS_URL)

    articles = []

    for entry in feed.entries[:max_results]:

        articles.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published_at": entry.get("published", ""),
        })

    return articles


# ==========================================
# 2-1. 여러 섹션 한 번에 가져오기 (대규모 기사 수집을 위한)
# ==========================================

SBS_SECTIONS = {
    "01": "정치",
    "02": "경제",
    "03": "사회",
    "07": "국제",
    "08": "생활문화",
    "09": "스포츠",
    "14": "연예",
}


def fetch_sbs_rss_multi(max_results_per_section=30):
    """
    SBS의 여러 섹션 RSS를 전부 가져와서 하나로 합친다.
    """

    all_articles = []

    for section_code, section_name in SBS_SECTIONS.items():

        url = f"https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId={section_code}&plink=RSSREADER"

        feed = feedparser.parse(url)

        print(f"  {section_name}(sectionId={section_code}): {len(feed.entries)}개")

        for entry in feed.entries[:max_results_per_section]:

            all_articles.append(
                {
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                    "section": section_name,
                }
            )

    return all_articles


# ==========================================
# 3. SBS 기사 원문 가져오기
# ==========================================

def fetch_sbs_article(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    article = soup.select_one(".text_area")

    if article is None:
        article = soup.select_one("#newsContent")

    if article is None:
        article = soup.select_one(".main_text")

    if article is None:
        return ""

    text = article.get_text("\n", strip=True)

    return text


# ==========================================
# 4. 테스트
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("SBS 원문 뉴스 수집 테스트")
    print("=" * 60)

    articles = fetch_sbs_rss(5)

    print(f"\nRSS 기사 {len(articles)}개 수집\n")

    for i, article in enumerate(articles, start=1):

        print("-" * 60)
        print(f"[{i}] {article['title']}")
        print(f"URL: {article['url']}")

        try:

            content = fetch_sbs_article(article["url"])

            print(f"\n원문 길이: {len(content)}자")

            print("\n원문 앞부분:")
            print(content[:500])

        except Exception as e:

            print(f"원문 수집 실패: {e}")