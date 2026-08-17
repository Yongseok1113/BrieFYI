"""검색 mode와 RRF/normalized hybrid 결합 테스트."""
import unittest
from unittest import mock

from rag.retriever import (
    _apply_metadata_boost,
    _combine_normalized_scores,
    _combine_rrf_scores,
    retrieve,
)


def row(
    chunk_id,
    vector_score=None,
    text_score=None,
    *,
    category=None,
    domains=None,
):
    value = {
        "article_id": chunk_id,
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "text": f"chunk {chunk_id}",
        "title": f"title {chunk_id}",
        "url": f"https://example.com/{chunk_id}",
        "category": category,
        "domains": domains or [],
        "entities": [],
    }
    if vector_score is not None:
        value["vector_score"] = vector_score
    if text_score is not None:
        value["text_score"] = text_score
    return value


class HybridTest(unittest.TestCase):
    def test_rrf는_검색기별_순위를_가중합한다(self):
        results = _combine_rrf_scores(
            vector_rows=[row(1, vector_score=0.9), row(2, vector_score=0.5)],
            text_rows=[row(2, text_score=10.0), row(3, text_score=5.0)],
            vector_weight=0.7,
            text_weight=0.3,
            top_k=3,
            rrf_k=60,
        )

        self.assertEqual([2, 1, 3], [result["chunk_id"] for result in results])
        self.assertEqual(2, results[0]["vector_rank"])
        self.assertEqual(1, results[0]["text_rank"])
        self.assertAlmostEqual(0.7 / 62 + 0.3 / 61, results[0]["score"])

    def test_rrf는_동일한_원점수에_같은_순위를_부여한다(self):
        results = _combine_rrf_scores(
            vector_rows=[],
            text_rows=[
                row(1, text_score=0.3),
                row(2, text_score=0.3),
                row(3, text_score=0.2),
            ],
            vector_weight=0.7,
            text_weight=0.3,
            top_k=3,
            rrf_k=60,
        )

        self.assertEqual([1, 1, 2], [result["text_rank"] for result in results])
        self.assertAlmostEqual(results[0]["text_rrf_score"], results[1]["text_rrf_score"])

    def test_normalized방식도_비교용으로_선택할수있다(self):
        results = _combine_normalized_scores(
            vector_rows=[row(1, vector_score=0.9), row(2, vector_score=0.5)],
            text_rows=[row(2, text_score=10.0), row(3, text_score=5.0)],
            vector_weight=0.7,
            text_weight=0.3,
            top_k=3,
        )

        self.assertEqual([1, 2, 3], [result["chunk_id"] for result in results])
        self.assertAlmostEqual(0.7, results[0]["score"])
        self.assertAlmostEqual(0.3, results[1]["score"])
        self.assertAlmostEqual(0.15, results[2]["score"])

    @mock.patch("rag.retriever._text_search")
    @mock.patch("rag.retriever._vector_search")
    def test_hybrid는_두_검색을_호출한다(self, vector_search, text_search):
        vector_search.return_value = [row(1, vector_score=0.8)]
        text_search.return_value = [row(1, text_score=2.0)]

        results = retrieve("query", top_k=5, search_mode="hybrid")

        self.assertEqual(1, results[0]["chunk_id"])
        self.assertAlmostEqual(1 / 61, results[0]["score"])
        vector_search.assert_called_once_with("query", 50, "cosine")
        text_search.assert_called_once_with("query", 50)

    def test_잘못된_가중치를_거부한다(self):
        with self.assertRaises(ValueError):
            _combine_rrf_scores([], [], 0, 0, 5, 60)

    def test_candidate_k는_top_k보다_작을수없다(self):
        with self.assertRaises(ValueError):
            retrieve("query", top_k=5, search_mode="hybrid", candidate_k=4)

    def test_Category_Domain_일치는_후보를_제거하지_않고_가산점을_준다(self):
        rows = [
            {**row(1, category="경제", domains=["증시"]), "score": 1.0},
            {**row(2, category="기술", domains=["AI"]), "score": 0.96},
        ]

        results = _apply_metadata_boost(
            rows,
            category="기술",
            domains=["AI"],
            category_boost=0.05,
            domain_boost=0.05,
            top_k=2,
        )

        self.assertEqual([2, 1], [result["chunk_id"] for result in results])
        self.assertAlmostEqual(0.96, results[0]["base_score"])
        self.assertAlmostEqual(0.1, results[0]["metadata_score"])
        self.assertTrue(results[0]["category_match"])
        self.assertEqual(["AI"], results[0]["matched_domains"])

    @mock.patch("rag.retriever._vector_search")
    def test_metadata_가산점이_있으면_vector_후보를_더_가져온다(self, vector_search):
        vector_search.return_value = [row(1, vector_score=0.8)]

        retrieve("query", top_k=3, search_mode="vector", category="기술")

        vector_search.assert_called_once_with("query", 50, "cosine")

    def test_metadata_가산점은_음수일수없다(self):
        with self.assertRaises(ValueError):
            retrieve("query", category="기술", category_boost=-0.1)


if __name__ == "__main__":
    unittest.main()
