import unittest
from unittest.mock import patch

from rag_latest.llm_client import LLMError, call_llm, parse_json_response


class ParseJsonResponseTest(unittest.TestCase):
    def test_코드블록으로_감싼_JSON을_파싱한다(self):
        text = '설명입니다\n```json\n{"score": 85}\n```\n'
        self.assertEqual(parse_json_response(text), {"score": 85})

    def test_코드블록_없이_바로_JSON이면_그대로_파싱한다(self):
        self.assertEqual(parse_json_response('{"passed": true}'), {"passed": True})

    def test_JSON_뒤에_부연설명이_붙어도_첫_JSON만_파싱한다(self):
        text = '{"summary": "x"} 이상입니다.'
        self.assertEqual(parse_json_response(text), {"summary": "x"})


class CallLlmProviderRoutingTest(unittest.TestCase):
    def test_지원하지_않는_provider면_LLMError(self):
        with self.assertRaises(LLMError):
            call_llm("sys", "user", provider="does-not-exist")

    def test_provider_생략하면_config_기본값을_쓴다(self):
        with patch("rag_latest.llm_client.config") as mock_config, \
             patch("rag_latest.llm_client._call_groq", return_value="ok") as mock_groq:
            mock_config.RAG_SUMMARIZE_PROVIDER = "groq"
            result = call_llm("sys", "user")
        mock_groq.assert_called_once()
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
