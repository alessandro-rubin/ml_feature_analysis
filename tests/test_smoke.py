def test_import():
    import tessa

    assert tessa.__version__


def test_config_defaults():
    from tessa import Config

    cfg = Config()
    assert cfg.timestamp_col == "timestamp"
    assert cfg.asset_dir("A1").as_posix() == "data/A1"
    assert Config(asset_subdir="input").asset_dir("A1").as_posix() == "data/A1/input"
