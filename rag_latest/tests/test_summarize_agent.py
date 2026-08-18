import unittest
from unittest.mock import patch

from rag_latest.summarize_agent import summarize_with_verification

CHUNKS = [{"article_id": 1, "title": "제목", "url": "https://x/1", "text": "본문"}]


def _llm_responses(*json_strings):
    """call_llm이 호출 순서대로 반환할 문자열 시퀀스. 한 시도당 summarize->grounding->judge
    3번 호출되므로, 시도 N개면 길이 3N짜리 시퀀스를 넘긴다."""
    return list(json_strings)


class SummarizeWithVerificationTest(unittest.TestCase):
    def test_1회차에_바로_통과하면_1개_시도만_기록된다(self):
        responses = _llm_responses(
            '{"summary": "요약1", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 85, "reasoning": "좋음", "breakdown": {}}',
        )
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses) as mock_call, \
             patch("rag_latest.summarize_agent.save_agent_run") as mock_save:
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(mock_call.call_count, 3)
        mock_save.assert_called_once()

    def test_1회차_미달_2회차_통과(self):
        responses = _llm_responses(
            '{"summary": "요약1", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 50, "reasoning": "부족", "breakdown": {}}',
            '{"summary": "요약2", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 90, "reasoning": "좋음", "breakdown": {}}',
        )
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses), \
             patch("rag_latest.summarize_agent.save_agent_run") as mock_save:
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.final.judge_score, 90)
        self.assertEqual(mock_save.call_count, 2)

    def test_max_attempts_소진하면_실패_플래그로_반환한다(self):
        one_attempt = [
            '{"summary": "요약", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 40, "reasoning": "부족", "breakdown": {}}',
        ]
        responses = one_attempt * 2  # max_attempts=2
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses):
            result = summarize_with_verification("질의", CHUNKS, max_attempts=2, score_threshold=70)

        self.assertFalse(result.passed)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.final.passed_threshold)

    def test_grounding_실패하면_점수가_높아도_재시도한다(self):
        responses = _llm_responses(
            '{"summary": "요약1", "citations": []}',
            '{"passed": false, "issues": [{"sentence": "x", "reason": "근거 없음"}]}',
            '{"score": 99, "reasoning": "점수는 높음", "breakdown": {}}',
            '{"summary": "요약2", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 80, "reasoning": "좋음", "breakdown": {}}',
        )
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses), \
             patch("rag_latest.summarize_agent.save_agent_run"):
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertTrue(result.passed)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].passed_threshold)  # grounding 실패 + 점수 99여도 미달 처리

    def test_빈_chunk는_LLM_호출_없이_즉시_실패_반환한다(self):
        with patch("rag_latest.summarize_agent.call_llm") as mock_call:
            result = summarize_with_verification("질의", [], max_attempts=3, score_threshold=70)

        mock_call.assert_not_called()
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, [])

    def test_JSON_파싱_실패해도_루프가_죽지_않고_다음_시도로_넘어간다(self):
        responses = [
            "이건 JSON이 아님",  # summarize 파싱 실패
            '{"summary": "요약2", "citations": []}',
            '{"passed": true, "issues": []}',
            '{"score": 90, "reasoning": "좋음", "breakdown": {}}',
        ]
        with patch("rag_latest.summarize_agent.call_llm", side_effect=responses), \
             patch("rag_latest.summarize_agent.save_agent_run"):
            result = summarize_with_verification("질의", CHUNKS, max_attempts=3, score_threshold=70)

        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].passed_threshold)
        self.assertEqual(result.attempts[0].judge_score, 0)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
