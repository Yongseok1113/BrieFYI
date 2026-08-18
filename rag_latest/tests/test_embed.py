"""실제 HF 호출 없이 embedding 응답 계약을 검증한다."""
import unittest
from unittest import mock

from rag_latest.embed import embed_query, embed_texts


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class EmbedTest(unittest.TestCase):
    @mock.patch("rag_latest.embed.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag_latest.embed.config.HF_TOKEN", "test-token")
    @mock.patch("rag_latest.embed.requests.post")
    def test_여러_문장을_정규화한다(self, post):
        post.return_value = FakeResponse([[3, 4, 0], [0, 0, 2]])

        vectors = embed_texts(["first", "second"])

        self.assertEqual([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], vectors)
        self.assertEqual(["first", "second"], post.call_args.kwargs["json"]["inputs"])

    @mock.patch("rag_latest.embed.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag_latest.embed.config.HF_TOKEN", "test-token")
    @mock.patch("rag_latest.embed.requests.post")
    def test_단일_flat_vector도_허용한다(self, post):
        post.return_value = FakeResponse([0, 3, 4])
        self.assertEqual([0.0, 0.6, 0.8], embed_query("query"))

    @mock.patch("rag_latest.embed.config.HF_TOKEN", "")
    def test_HF_TOKEN이_없으면_호출하지_않는다(self):
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            embed_texts(["text"])


if __name__ == "__main__":
    unittest.main()
