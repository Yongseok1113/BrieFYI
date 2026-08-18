import tempfile
from pathlib import Path
from unittest.mock import patch

from make_train_data.cli import run


def test_기사가_MIN_ARTICLES_미만이면_1을_반환하고_export하지_않는다():
    with patch("make_train_data.cli.fetch_articles", return_value=[{"id": 1}] * 5), \
         patch("make_train_data.cli.config") as mock_config, \
         patch("make_train_data.cli.export_clusters") as mock_export:
        mock_config.MTD_MIN_ARTICLES = 20
        with tempfile.TemporaryDirectory() as tmp:
            code = run(out_dir=Path(tmp))
    assert code == 1
    mock_export.assert_not_called()


def test_기사가_충분하면_전체_파이프라인을_거쳐_export한다():
    articles = [{"id": i} for i in range(25)]
    with patch("make_train_data.cli.fetch_articles", return_value=articles), \
         patch("make_train_data.cli.config") as mock_config, \
         patch("make_train_data.cli.cluster_articles", return_value=([], articles)) as mock_cluster, \
         patch("make_train_data.cli.select_onefact_candidates", return_value=articles[:4]) as mock_onefact, \
         patch("make_train_data.cli.export_clusters", return_value=[Path("x.json")]) as mock_export:
        mock_config.MTD_MIN_ARTICLES = 20
        mock_config.MTD_NARROW_WINDOW_HOURS = 72
        mock_config.MTD_BROAD_WINDOW_DAYS = 28
        mock_config.MTD_ENTITY_JACCARD_THRESHOLD = 0.3
        mock_config.MTD_EMBED_SIM_THRESHOLD = 0.75
        mock_config.MTD_DEDUP_THRESHOLD = 0.9
        mock_config.MTD_MIN_CLUSTER_SIZE = 2
        mock_config.MTD_ONEFACT_RATIO = 0.175
        with tempfile.TemporaryDirectory() as tmp:
            code = run(out_dir=Path(tmp))
    assert code == 0
    mock_cluster.assert_called_once()
    mock_onefact.assert_called_once()
    mock_export.assert_called_once()
