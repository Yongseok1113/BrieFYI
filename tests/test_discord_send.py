import unittest
from unittest.mock import patch

from tools.discord_send import send_discord


class DiscordWebhookConfigTest(unittest.TestCase):
    def test_웹훅_URL이_없으면_RuntimeError를_던진다(self):
        with patch("tools.discord_send.config") as mock_config:
            mock_config.DISCORD_WEBHOOK_URL = ""
            with self.assertRaises(RuntimeError) as ctx:
                send_discord("제목", summaries=[], insight={})
        self.assertIn("DISCORD_WEBHOOK_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
