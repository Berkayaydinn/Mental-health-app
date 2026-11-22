import os

from inference.predict import recommend


def test_relative_model_dir_works_from_any_cwd(tmp_path):
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        resp = recommend("u1", k=3, model_dir="models/recommendation/latest")
    finally:
        os.chdir(old)
    assert resp.user_id == "u1"
    assert resp.results
