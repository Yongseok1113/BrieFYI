# tests/test_distributor.py
import unittest
from unittest.mock import MagicMock

from agents.distributor import DistributorAgent


def _make_state(digest_id=1):
    return {
        "digest_date": "2026-08-18",
        "keyword": "AI",
        "email_html": "<p>본문</p>",
        "digest_id": digest_id,
    }


class DistributorMultiRecipientTest(unittest.TestCase):
    def test_수신자별로_개별_발송하고_개별_로그를_남긴다(self):
        email_tool = MagicMock(return_value={"id": "sent"})
        log_send_tool = MagicMock()
        agent = DistributorAgent(
            name="distributor",
            tools={"email": email_tool, "log_send": log_send_tool},
            channels=["email"],
        )

        with unittest.mock.patch("agents.distributor.config") as mock_config:
            mock_config.EMAIL_RECIPIENTS = ["a@example.com", "b@example.com"]
            result = agent.run(_make_state())

        self.assertEqual(email_tool.call_count, 2)
        email_tool.assert_any_call(unittest.mock.ANY, "<p>본문</p>", to="a@example.com")
        email_tool.assert_any_call(unittest.mock.ANY, "<p>본문</p>", to="b@example.com")
        self.assertEqual(log_send_tool.call_count, 2)
        self.assertEqual(result["send_result"]["success_count"], 2)
        self.assertEqual(result["send_result"]["total_count"], 2)

    def test_일부_수신자_발송_실패해도_나머지는_계속_보낸다(self):
        email_tool = MagicMock(side_effect=[RuntimeError("발송 실패"), {"id": "sent"}])
        log_send_tool = MagicMock()
        agent = DistributorAgent(
            name="distributor",
            tools={"email": email_tool, "log_send": log_send_tool},
            channels=["email"],
        )

        with unittest.mock.patch("agents.distributor.config") as mock_config:
            mock_config.EMAIL_RECIPIENTS = ["fail@example.com", "ok@example.com"]
            result = agent.run(_make_state())

        self.assertEqual(result["send_result"]["success_count"], 1)
        self.assertEqual(result["send_result"]["total_count"], 2)
        log_send_tool.assert_any_call(1, "email", "fail@example.com", "failed", "발송 실패")
        log_send_tool.assert_any_call(1, "email", "ok@example.com", "success")

    def test_email이_channels에_없으면_아무것도_하지_않는다(self):
        email_tool = MagicMock()
        agent = DistributorAgent(
            name="distributor",
            tools={"email": email_tool, "log_send": MagicMock()},
            channels=[],
        )
        result = agent.run(_make_state())
        email_tool.assert_not_called()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
