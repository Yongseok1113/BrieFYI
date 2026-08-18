"""RAG 텍스트 구성과 overlap 청킹 테스트."""
import unittest

from rag.chunk import build_article_text, split_text


class OffsetTokenizer:
    def __call__(self, text, **kwargs):
        offsets = []
        cursor = 0
        for token in text.split(" "):
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"offset_mapping": offsets}


class ChunkTest(unittest.TestCase):
    def test_title과_description을_결합한다(self):
        self.assertEqual("제목\n\n설명", build_article_text(" 제목 ", " 설명 "))

    def test_tokenizer가_없으면_기사_전체가_한_청크다(self):
        self.assertEqual(
            [{"chunk_index": 0, "chunk_text": "one two three"}],
            split_text("one two three"),
        )

    def test_overlap_청킹은_구현되어_있다(self):
        chunks = split_text(
            "one two three four five six",
            tokenizer=OffsetTokenizer(),
            chunk_size=4,
            overlap=2,
        )
        self.assertEqual(["one two three four", "three four five six"], [c["chunk_text"] for c in chunks])
        self.assertEqual([0, 1], [c["chunk_index"] for c in chunks])

    def test_overlap은_chunk_size보다_작아야_한다(self):
        with self.assertRaises(ValueError):
            split_text("text", tokenizer=OffsetTokenizer(), chunk_size=2, overlap=2)


if __name__ == "__main__":
    unittest.main()
