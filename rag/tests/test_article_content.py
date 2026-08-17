"""실제 외부 요청 없이 기사 본문 HTML 추출 규칙을 검증한다."""
import importlib.util
import unittest
from unittest import mock

import requests

from tools.article_content import fetch_article_body


BS4_AVAILABLE = importlib.util.find_spec("bs4") is not None


def _response(html: str, content_type: str = "text/html; charset=utf-8"):
    response = mock.Mock()
    response.text = html
    response.headers = {"Content-Type": content_type}
    response.raise_for_status.return_value = None
    return response


@unittest.skipUnless(BS4_AVAILABLE, "beautifulsoup4가 설치된 환경에서 실행")
class ArticleContentTest(unittest.TestCase):
    @mock.patch("tools.article_content.requests.get")
    def test_article을_main보다_우선하고_짧은_본문도_사용한다(self, get):
        get.return_value = _response(
            """<html><body>
            <main><p>main fallback text</p></main>
            <article><script>remove me</script><p>짧은 본문</p></article>
            </body></html>"""
        )

        body = fetch_article_body("https://example.com/news")

        self.assertEqual("짧은 본문", body)
        get.assert_called_once_with(
            "https://example.com/news",
            headers=mock.ANY,
            timeout=15,
        )

    @mock.patch("tools.article_content.requests.get")
    def test_전체_p_fallback에서는_40자_이하를_제외한다(self, get):
        long_paragraph = "이 문단은 페이지 전체 fallback에서도 기사 본문으로 남을 만큼 충분히 긴 문장입니다."
        get.return_value = _response(
            f"<html><body><p>짧은 메뉴</p><p>{long_paragraph}</p></body></html>"
        )

        self.assertEqual(
            long_paragraph,
            fetch_article_body("https://example.com/fallback"),
        )

    @mock.patch("tools.article_content.requests.get")
    def test_본문이_완전히_비면_실패한다(self, get):
        get.return_value = _response("<html><body><p>짧음</p></body></html>")

        with self.assertRaisesRegex(ValueError, "본문"):
            fetch_article_body("https://example.com/empty")

    @mock.patch("tools.article_content.requests.get")
    def test_HTML이_아니면_거부한다(self, get):
        get.return_value = _response("binary", "application/pdf")

        with self.assertRaisesRegex(ValueError, "HTML"):
            fetch_article_body("https://example.com/file.pdf")

    @mock.patch(
        "tools.article_content.requests.get",
        side_effect=requests.Timeout("timeout"),
    )
    def test_요청_오류를_호출자에게_전달한다(self, _get):
        with self.assertRaises(requests.Timeout):
            fetch_article_body("https://example.com/timeout")


if __name__ == "__main__":
    unittest.main()

