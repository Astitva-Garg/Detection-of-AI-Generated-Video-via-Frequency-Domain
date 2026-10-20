from src import config


def test_constants():
    assert config.N_FRAMES == 16 and config.FPS == 8 and config.N_BINS == 48
    assert set(config.SEEN).isdisjoint(config.UNSEEN)


def test_ensure_dirs(tmp_path):
    config.ensure_dirs(tmp_path)
    for p in config.paths(tmp_path).values():
        assert p.is_dir()
