from app.config import Settings, get_settings


def test_defaults_match_spec():
    s = Settings()
    assert s.MODEL_ID == "MBZUAI/AIN"
    assert s.TORCH_DTYPE == "float16"
    assert s.LOAD_IN_8BIT is True
    assert s.MAX_NEW_TOKENS == 320
    assert s.SELF_CONSISTENCY is True
    assert s.SHOW_BOXES is False
    assert s.DEBUG_SAVE_IMAGES is False


def test_env_overrides_model_id(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    assert Settings().MODEL_ID == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_show_boxes_can_be_enabled_by_env(monkeypatch):
    monkeypatch.setenv("SHOW_BOXES", "true")
    assert Settings().SHOW_BOXES is True


def test_load_in_8bit_can_be_disabled_by_env(monkeypatch):
    # The fallback path (fp16, no quantisation - e.g. switching back to the
    # 3B Qwen2.5-VL model) must be reachable without editing code.
    monkeypatch.setenv("LOAD_IN_8BIT", "false")
    assert Settings().LOAD_IN_8BIT is False


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
