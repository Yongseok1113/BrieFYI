"""실제 외부 요청 없이 본문 추출 규칙과 overlap 청킹을 검증한다."""
import importlib.util
import unittest
from unittest import mock

import requests

from rag.content import build_article_text, fetch_article_body, split_text

BS4_AVAILABLE = importlib.util.find_spec("bs4") is not None


class OffsetTokenizer:
    """offset mapping만 흉내 내는 테스트용 fast tokenizer."""

    def __call__(self, text, **kwargs):
        offsets = []
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"offset_mapping": offsets}


def _response(html: str, content_type: str = "text/html; charset=utf-8"):
    response = mock.Mock()
    response.text = html
    response.headers = {"Content-Type": content_type}
    response.raise_for_status.return_value = None
    return response


@unittest.skipUnless(BS4_AVAILABLE, "beautifulsoup4가 설치된 환경에서 실행")
class ArticleBodyTest(unittest.TestCase):
    @mock.patch("rag.content.requests.get")
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

    @mock.patch("rag.content.requests.get")
    def test_전체_p_fallback에서는_40자_이하를_제외한다(self, get):
        long_paragraph = "이 문단은 페이지 전체 fallback에서도 기사 본문으로 남을 만큼 충분히 긴 문장입니다."
        get.return_value = _response(
            f"<html><body><p>짧은 메뉴</p><p>{long_paragraph}</p></body></html>"
        )

        self.assertEqual(
            long_paragraph,
            fetch_article_body("https://example.com/fallback"),
        )

    @mock.patch("rag.content.requests.get")
    def test_본문이_완전히_비면_실패한다(self, get):
        get.return_value = _response("<html><body><p>짧음</p></body></html>")

        with self.assertRaisesRegex(ValueError, "본문"):
            fetch_article_body("https://example.com/empty")

    @mock.patch("rag.content.requests.get")
    def test_HTML이_아니면_거부한다(self, get):
        get.return_value = _response("binary", "application/pdf")

        with self.assertRaisesRegex(ValueError, "HTML"):
            fetch_article_body("https://example.com/file.pdf")

    @mock.patch(
        "rag.content.requests.get",
        side_effect=requests.Timeout("timeout"),
    )
    def test_요청_오류를_호출자에게_전달한다(self, _get):
        with self.assertRaises(requests.Timeout):
            fetch_article_body("https://example.com/timeout")


class ChunkTest(unittest.TestCase):
    def test_title과_description을_결합한다(self):
        self.assertEqual("제목\n\n설명", build_article_text(" 제목 ", " 설명 "))

    def test_본문이_있으면_description_대신_사용한다(self):
        self.assertEqual(
            "제목\n\n실제 본문",
            build_article_text("제목", "짧은 설명", " 실제 본문 "),
        )

    def test_overlap_청킹은_구현되어_있다(self):
        chunks = split_text(
            "one two three four five six",
            OffsetTokenizer(),
            chunk_size=4,
            overlap=2,
        )
        self.assertEqual(
            ["one two three four", "three four five six"],
            [chunk["chunk_text"] for chunk in chunks],
        )
        self.assertEqual([0, 1], [chunk["chunk_index"] for chunk in chunks])

    def test_overlap은_chunk_size보다_작아야_한다(self):
        with self.assertRaises(ValueError):
            split_text("text", OffsetTokenizer(), chunk_size=2, overlap=2)

    def test_기본값은_500_token과_50_token_overlap이다(self):
        tokens = [f"t{i}" for i in range(600)]
        chunks = split_text(" ".join(tokens), OffsetTokenizer())
        chunk_tokens = [chunk["chunk_text"].split() for chunk in chunks]

        self.assertEqual([500, 150], [len(items) for items in chunk_tokens])
        self.assertEqual(chunk_tokens[0][-50:], chunk_tokens[1][:50])


if __name__ == "__main__":
    unittest.main()
