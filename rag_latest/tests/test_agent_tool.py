import unittest
from unittest.mock import patch

from rag_latest.agent_tool import SEARCH_NEWS_TOOL_SCHEMA, execute_search_news, search_news


class SearchNewsTest(unittest.TestCase):
    def test_retrieve와_rerank를_거쳐_직렬화된_결과를_반환한다(self):
        fake_rows = [
            {
                "article_id": 1,
                "chunk_id": 10,
                "title": "제목",
                "url": "https://example.com/1",
                "text": "x" * 600,
                "category": "기술",
                "domains": ["AI"],
                "score": 0.5,
            }
        ]
        reranked_rows = [{**fake_rows[0], "rerank_score": 0.987654}]

        with patch("rag_latest.agent_tool.retriever.retrieve", return_value=fake_rows) as mock_retrieve, \
             patch("rag_latest.agent_tool.reranker.rerank", return_value=reranked_rows) as mock_rerank, \
             patch("rag_latest.agent_tool.reranker.unload_reranker") as mock_unload:
            result = search_news("Claude 관련 뉴스", top_k=5)

        mock_retrieve.assert_called_once()
        mock_rerank.assert_called_once()
        mock_unload.assert_called_once()
        self.assertEqual(result[0]["article_id"], 1)
        self.assertEqual(len(result[0]["text"]), 500)
        self.assertEqual(result[0]["rerank_score"], 0.9877)

    def test_execute_search_news는_dict_인자를_search_news로_그대로_전달한다(self):
        with patch("rag_latest.agent_tool.search_news", return_value=[]) as mock_search:
            execute_search_news({"query": "AI", "top_k": 3, "category": "기술"})

        mock_search.assert_called_once_with(query="AI", top_k=3, category="기술", domains=None)

    def test_tool_schema의_category_enum은_taxonomy와_일치한다(self):
        from rag_latest import taxonomy

        enum_values = SEARCH_NEWS_TOOL_SCHEMA["input_schema"]["properties"]["category"]["enum"]
        self.assertEqual(enum_values, list(taxonomy.CATEGORY_LABELS))


if __name__ == "__main__":
    unittest.main()
