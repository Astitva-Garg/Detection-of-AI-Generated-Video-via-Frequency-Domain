from src.data.content_filter import data_key


def test_same_key_wherever_the_data_folder_is():
    old_abs = r"C:\ie643_data\raw\real\12751033.mp4"
    new_abs = r"C:\ie643_project\data\raw\real\12751033.mp4"
    posix = "/home/user/data/raw/real/12751033.mp4"
    rel = "raw/real/12751033.mp4"
    assert data_key(old_abs) == data_key(new_abs) == data_key(posix) == data_key(rel) == rel


def test_generated_clip_keys_keep_the_generator_folder():
    assert data_key(r"D:\x\raw\fake\wan\wan_0003.mp4") == "raw/fake/wan/wan_0003.mp4"
