import unittest
from unittest.mock import patch

from rag_latest.reranker import rerank


class RerankTest(unittest.TestCase):
    def test_후보를_cross_encoder_점수_순으로_재정렬한다(self):
        rows = [
            {"chunk_id": 1, "title": "A", "text": "text a", "score": 0.9},
            {"chunk_id": 2, "title": "B", "text": "text b", "score": 0.8},
        ]
        fake_reranker = type(
            "FakeReranker", (), {"compute_score": staticmethod(lambda pairs, normalize=True: [0.1, 0.9])}
        )()
        with patch("rag_latest.reranker.get_reranker", return_value=fake_reranker):
            result = rerank("query", rows)

        self.assertEqual([row["chunk_id"] for row in result], [2, 1])
        self.assertAlmostEqual(result[0]["rerank_score"], 0.9)
        self.assertEqual(result[0]["pre_rerank_rank"], 2)

    def test_빈_후보는_그대로_빈_리스트를_반환한다(self):
        self.assertEqual(rerank("query", []), [])

    def test_top_k로_결과를_자른다(self):
        rows = [{"chunk_id": i, "title": "t", "text": "x", "score": 0.0} for i in range(5)]
        fake_reranker = type(
            "FakeReranker", (), {"compute_score": staticmethod(lambda pairs, normalize=True: [0.5] * 5)}
        )()
        with patch("rag_latest.reranker.get_reranker", return_value=fake_reranker):
            result = rerank("query", rows, top_k=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
