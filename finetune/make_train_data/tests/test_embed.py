from unittest.mock import patch

from make_train_data.embed import embed_texts


def test_rag_latest_embed_texts를_그대로_호출한다():
    with patch("make_train_data.embed._embed_texts", return_value=[[0.1, 0.2]]) as mock_embed:
        result = embed_texts(["hello"])
    mock_embed.assert_called_once_with(["hello"])
    assert result == [[0.1, 0.2]]
