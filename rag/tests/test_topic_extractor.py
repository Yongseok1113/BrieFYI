"""실제 GLiNER2 model 없이 Category/Domain/Entity 집계 계약을 검증한다."""
import unittest
from unittest import mock

from rag.topic_extractor import GLiNER2TopicExtractor


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        return text.split()


class FakeSchema:
    def entities(self, *_args, **_kwargs):
        return self

    def classification(self, *_args, **_kwargs):
        return self


class FakeModel:
    def __init__(self, results):
        self.processor = mock.Mock(tokenizer=FakeTokenizer())
        self.results = iter(results)

    def create_schema(self):
        return FakeSchema()

    def extract(self, *_args, **_kwargs):
        return next(self.results)


class TopicExtractorTest(unittest.TestCase):
    def test_여러_window의_metadata를_article_level로_집계한다(self):
        model = FakeModel(
            [
                {
                    "category": {"label": "기술", "confidence": 0.8},
                    "domain": [
                        {"label": "AI", "confidence": 0.9},
                        {"label": "기타", "confidence": 0.5},
                    ],
                    "entities": {
                        "company": [
                            {"text": "NVIDIA가", "confidence": 0.9},
                            {"text": "OpenAI", "confidence": 0.8},
                        ]
                    },
                },
                {
                    "category": {"label": "산업", "confidence": 0.95},
                    "domain": [
                        {"label": "반도체", "confidence": 0.85},
                        {"label": "AI", "confidence": 0.7},
                    ],
                    "entities": {
                        "company": [
                            {"text": "NVIDIA", "confidence": 0.7},
                            {"text": "Blackwell", "confidence": 0.96},
                            {"text": "Microsoft", "confidence": 0.4},
                        ]
                    },
                },
            ]
        )
        extractor = GLiNER2TopicExtractor(model)

        result = extractor.extract(" ".join(f"word{i}" for i in range(450)))

        self.assertEqual("산업", result["category"])
        self.assertEqual(["AI", "반도체"], result["domains"])
        self.assertEqual(["Blackwell", "NVIDIA", "OpenAI"], result["entities"])
        self.assertEqual("business_tech_v1", result["topic_taxonomy_version"])

    def test_분류_출력이_없으면_기타를_사용한다(self):
        extractor = GLiNER2TopicExtractor(FakeModel([{"entities": {}}]))

        result = extractor.extract("short text")

        self.assertEqual("기타", result["category"])
        self.assertEqual(["기타"], result["domains"])
        self.assertEqual([], result["entities"])


if __name__ == "__main__":
    unittest.main()

