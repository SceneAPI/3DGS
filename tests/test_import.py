import sceneapi_3dgs


def test_import_exposes_version() -> None:
    assert isinstance(sceneapi_3dgs.__version__, str)
