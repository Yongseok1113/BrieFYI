"""외부 API와 DB 없이 GNews 수집→4-Layer indexing 연결 계약을 검증한다."""
import unittest
from unittest import mock

from rag.ingestion import collect_and_index


class IngestionTest(unittest.TestCase):
    @mock.patch("rag.ingestion.index_articles")
    @mock.patch("rag.ingestion._article_ids_for_urls", side_effect=[[], [11, 12]])
    @mock.patch("rag.ingestion.insert_articles", return_value=2)
    @mock.patch("rag.ingestion.fetch_news")
    def test_이번에_수집한_URL의_기사를_GLiNER2_indexer로_넘긴다(
        self,
        fetch_news,
        insert_articles,
        article_ids_for_urls,
        index_articles,
    ):
        fetch_news.return_value = [
            {"title": "A", "url": "https://example.com/a"},
            {"title": "B", "url": "https://example.com/b"},
        ]
        index_articles.return_value = [{"article_id": 11}, {"article_id": 12}]

        result = collect_and_index("AI", 2, 5, device="cpu")

        fetch_news.assert_called_once_with("AI", 2, 5)
        self.assertEqual(2, len(insert_articles.call_args.args[1]))
        self.assertEqual(
            [
                mock.call(["https://example.com/a", "https://example.com/b"]),
                mock.call(["https://example.com/a", "https://example.com/b"]),
            ],
            article_ids_for_urls.call_args_list,
        )
        index_articles.assert_called_once_with([11, 12], device="cpu")
        self.assertEqual(2, result["inserted_count"])
        self.assertEqual([11, 12], result["article_ids"])

    @mock.patch("rag.ingestion.index_articles", return_value=[])
    @mock.patch("rag.ingestion._article_ids_for_urls", side_effect=[[7], [7]])
    @mock.patch("rag.ingestion.insert_articles", return_value=0)
    @mock.patch(
        "rag.ingestion.fetch_news",
        return_value=[{"title": "기존 기사", "url": "https://example.com/existing"}],
    )
    def test_이미_저장된_URL은_다시_indexing하지_않는다(
        self,
        _fetch_news,
        _insert_articles,
        _article_ids_for_urls,
        index_articles,
    ):
        result = collect_and_index("AI", device="cpu")

        index_articles.assert_called_once_with([], device="cpu")
        self.assertEqual([], result["article_ids"])
        self.assertEqual([], result["indexed"])


if __name__ == "__main__":
    unittest.main()
