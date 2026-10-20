import pandas as pd

from src.data.pexels import cap_per_scene, load_scenes, select_titles


def _df(rows):
    return pd.DataFrame(rows, columns=["title", "video", "duration", "sfw"])


def test_select_titles_keeps_and_drops():
    scenes = load_scenes()
    df = _df([
        ["Ocean Waves Crashing · Free Stock Video", "u1", 20, True],
        ["Man Walking on Beach at Sunset · Free Stock Video", "u2", 20, True],
        ["Ocean Waves Crashing on Rocks · Free Stock Video", "u3", 3, True],
        ["Boat on the Lake · Free Stock Video", "u4", 20, True],
        ["Waterfall in Forest · Free Stock Video", "u5", 20, False],
    ])
    out = select_titles(df, scenes)
    assert list(out["video"]) == ["u1"]
    assert out.iloc[0]["scene"] == "ocean waves"


def test_cap_per_scene():
    df = pd.DataFrame({"scene": ["a"] * 10 + ["b"] * 3, "view_count": range(13)})
    out = cap_per_scene(df, 4)
    assert (out["scene"] == "a").sum() == 4 and (out["scene"] == "b").sum() == 3
