import unittest

from rag_latest.prompts import (
    build_grounding_user_prompt,
    build_judge_user_prompt,
    build_summarize_user_prompt,
)


class BuildPromptsTest(unittest.TestCase):
    def test_summarize_prompt는_질의와_chunk를_포함한다(self):
        chunks = [{"article_id": 1, "title": "제목1", "url": "https://x/1", "text": "본문1", "category": "기술"}]
        prompt = build_summarize_user_prompt("Claude 관련 뉴스", chunks)
        self.assertIn("Claude 관련 뉴스", prompt)
        self.assertIn("제목1", prompt)
        self.assertIn("본문1", prompt)
        self.assertIn("1", prompt)  # article_id 인용 가능하도록 노출

    def test_grounding_prompt는_요약과_chunk를_포함한다(self):
        chunks = [{"article_id": 1, "title": "제목1", "text": "본문1"}]
        prompt = build_grounding_user_prompt("이것은 요약입니다", chunks)
        self.assertIn("이것은 요약입니다", prompt)
        self.assertIn("본문1", prompt)

    def test_judge_prompt는_질의_요약_grounding결과를_포함한다(self):
        grounding_result = {"passed": True, "issues": []}
        prompt = build_judge_user_prompt("질의", "요약문", grounding_result)
        self.assertIn("질의", prompt)
        self.assertIn("요약문", prompt)
        self.assertIn("true", prompt.lower())

    def test_빈_chunk_목록도_에러_없이_처리한다(self):
        prompt = build_summarize_user_prompt("질의", [])
        self.assertIn("질의", prompt)


if __name__ == "__main__":
    unittest.main()
