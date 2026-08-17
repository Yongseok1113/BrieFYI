"""실제 GLiNER2 model 없이 구조화 Event 후처리를 검증한다."""
import unittest

from rag.event_extractor import (
    GLiNER2EventExtractor,
    aggregate_event_candidates,
    event_candidates_from_result,
    normalize_argument_text,
    split_text_windows,
)


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class _FakeSchema:
    def entities(self, entity_types):
        self.entity_types = entity_types
        return self

    def relations(self, relation_types, threshold=None):
        self.relation_types = relation_types
        self.relation_threshold = threshold
        return self


class _FakeProcessor:
    tokenizer = _WordTokenizer()


class _FakeModel:
    processor = _FakeProcessor()

    def __init__(self, result):
        self.result = result
        self.schemas = []
        self.extract_calls = []

    def create_schema(self):
        schema = _FakeSchema()
        self.schemas.append(schema)
        return schema

    def extract(self, text, schema, **kwargs):
        self.extract_calls.append((text, schema, kwargs))
        return self.result


def _span(text: str, value: str, confidence: float = 0.9, *, start: int | None = None):
    start = text.index(value) if start is None else start
    return {
        "text": value,
        "confidence": confidence,
        "start": start,
        "end": start + len(value),
    }


class EventExtractorTest(unittest.TestCase):
    def test_GLiner_wrapper가_Entity_Relation과_span을_동시에_요청한다(self):
        text = "NVIDIA가 OpenAI에 투자했다."
        result = {
            "entities": {"company": [{"text": "NVIDIA"}, {"text": "OpenAI"}]},
            "relation_extraction": {
                "invested_in": [
                    {"head": _span(text, "NVIDIA가"), "tail": _span(text, "OpenAI에")}
                ]
            },
        }
        model = _FakeModel(result)

        events = GLiNER2EventExtractor(model).extract(text)

        self.assertEqual(1, len(events))
        self.assertIn("invested_in", model.schemas[1].relation_types)
        self.assertEqual(2, len(model.extract_calls))
        _text, _schema, kwargs = model.extract_calls[0]
        self.assertTrue(kwargs["include_confidence"])
        self.assertTrue(kwargs["include_spans"])

    def test_토큰_window가_overlap과_부모_offset을_보존한다(self):
        text = "one two three four five"

        windows = split_text_windows(
            text,
            _WordTokenizer(),
            max_tokens=3,
            overlap_words=1,
        )

        self.assertEqual(["one two three", "three four five"], [w["text"] for w in windows])
        self.assertEqual(
            [window["text"] for window in windows],
            [text[window["start"] : window["end"]] for window in windows],
        )

    def test_한국어_조사를_정리한다(self):
        self.assertEqual("NVIDIA", normalize_argument_text(" NVIDIA가 "))
        self.assertEqual("OpenAI", normalize_argument_text("OpenAI에"))

    def test_Entity와_일치하는_최고_confidence_relation만_남긴다(self):
        text = "NVIDIA가 OpenAI에 20억 달러를 투자했다."
        head = _span(text, "NVIDIA가", 0.94)
        tail = _span(text, "OpenAI에", 0.91)
        result = {
            "entities": {
                "company": [
                    {"text": "NVIDIA", "confidence": 0.95},
                    {"text": "OpenAI", "confidence": 0.93},
                ]
            },
            "relation_extraction": {
                "invested_in": [{"head": head, "tail": tail}],
                "acquired": [
                    {
                        "head": {**head, "confidence": 0.72},
                        "tail": {**tail, "confidence": 0.70},
                    }
                ],
            },
        }

        events = aggregate_event_candidates(event_candidates_from_result(result, text))

        self.assertEqual(1, len(events))
        self.assertEqual("invested_in", events[0]["event_type"])
        self.assertEqual(["investor", "investee"], [a["role"] for a in events[0]["arguments"]])
        self.assertEqual(["NVIDIA가", "OpenAI에"], [a["text"] for a in events[0]["arguments"]])
        self.assertEqual(["NVIDIA", "OpenAI"], [a["normalized_text"] for a in events[0]["arguments"]])
        self.assertAlmostEqual(0.91, events[0]["confidence"])

    def test_Entity가_아닌_argument와_같은_head_tail을_거부한다(self):
        text = "NVIDIA가 OpenAI에 투자했다."
        nvidia = _span(text, "NVIDIA가")
        openai = _span(text, "OpenAI에")
        result = {
            "entities": {"company": [{"text": "NVIDIA"}]},
            "relation_extraction": {
                "invested_in": [{"head": nvidia, "tail": openai}],
                "partnered_with": [{"head": nvidia, "tail": nvidia}],
            },
        }

        self.assertEqual([], event_candidates_from_result(result, text))

    def test_대칭_relation의_역방향과_overlap_중복을_제거한다(self):
        text = "Apple과 OpenAI가 협력했다."
        apple = _span(text, "Apple과", 0.86)
        openai = _span(text, "OpenAI가", 0.88)
        result = {
            "entities": {
                "company": [{"text": "Apple"}, {"text": "OpenAI"}],
            },
            "relation_extraction": {
                "partnered_with": [
                    {"head": apple, "tail": openai},
                    {
                        "head": {**openai, "confidence": 0.82},
                        "tail": {**apple, "confidence": 0.80},
                    },
                ]
            },
        }

        events = aggregate_event_candidates(event_candidates_from_result(result, text))

        self.assertEqual(1, len(events))
        self.assertEqual(["Apple", "OpenAI"], [a["normalized_text"] for a in events[0]["arguments"]])

    def test_임시_window_offset을_부모_chunk_span으로_복원한다(self):
        text = "NVIDIA가 OpenAI에 투자했다."
        result = {
            "entities": {"company": [{"text": "NVIDIA"}, {"text": "OpenAI"}]},
            "relation_extraction": {
                "invested_in": [
                    {"head": _span(text, "NVIDIA가"), "tail": _span(text, "OpenAI에")}
                ]
            },
        }

        event = event_candidates_from_result(result, text, source_offset=100)[0]

        self.assertEqual(100, event["arguments"][0]["span_start"])
        self.assertGreater(event["evidence_end"], 100)


if __name__ == "__main__":
    unittest.main()
