import unittest

from rag_latest.eval import recall_at_k, reciprocal_rank


class ScoringTest(unittest.TestCase):
    def test_reciprocal_rank는_target이_있는_위치의_역수를_반환한다(self):
        self.assertEqual(reciprocal_rank([10, 20, 30], 20), 0.5)
        self.assertEqual(reciprocal_rank([10, 20, 30], 10), 1.0)

    def test_reciprocal_rank는_target이_없으면_0을_반환한다(self):
        self.assertEqual(reciprocal_rank([10, 20, 30], 999), 0.0)

    def test_recall_at_k는_상위_k_안에_있는지_판정한다(self):
        self.assertTrue(recall_at_k([10, 20, 30, 40], 30, k=3))
        self.assertFalse(recall_at_k([10, 20, 30, 40], 40, k=3))


if __name__ == "__main__":
    unittest.main()
