import pytest
from PIL import Image

from app.model import FakeEngine, GenerationRequest, _check_model_format


def _img():
    return Image.new("RGB", (64, 64), "white")


def test_fake_engine_returns_scripted_responses_in_order():
    engine = FakeEngine(["first", "second"])
    out = engine.generate([
        GenerationRequest(image=_img(), prompt="a"),
        GenerationRequest(image=_img(), prompt="b"),
    ])
    assert out == ["first", "second"]


def test_fake_engine_records_calls_for_assertion():
    engine = FakeEngine(["x"])
    engine.generate([GenerationRequest(image=_img(), prompt="prompt-text")])
    assert len(engine.calls) == 1
    assert engine.calls[0][0].prompt == "prompt-text"


def test_fake_engine_accepts_a_callable_for_prompt_dependent_replies():
    engine = FakeEngine(lambda reqs: [r.prompt.upper() for r in reqs])
    assert engine.generate([GenerationRequest(image=_img(), prompt="hi")]) == ["HI"]


def test_fake_engine_raises_when_script_is_exhausted():
    engine = FakeEngine(["only-one"])
    engine.generate([GenerationRequest(image=_img(), prompt="a")])
    with pytest.raises(AssertionError):
        engine.generate([GenerationRequest(image=_img(), prompt="b")])


def test_fake_engine_reports_identity():
    engine = FakeEngine([])
    assert engine.model_id == "fake"
    assert engine.device == "cpu"


def test_gguf_repo_is_rejected_with_actionable_message():
    """GGUF repos have no HuggingFace config.json; transformers cannot load them.
    The raw error says only 'Unrecognized model', which does not help."""
    with pytest.raises(RuntimeError) as exc:
        _check_model_format("Qwen/Qwen3-VL-2B-Instruct-GGUF")
    msg = str(exc.value)
    assert "GGUF" in msg
    assert "Qwen/Qwen3-VL-2B-Instruct" in msg   # names the corrected repo


def test_mlx_repo_is_rejected():
    with pytest.raises(RuntimeError):
        _check_model_format("mlx-community/Qwen2-VL-7B-Instruct-4bit-MLX")


def test_a_normal_repo_passes():
    _check_model_format("Qwen/Qwen2.5-VL-3B-Instruct")
    _check_model_format("MBZUAI/AIN")


def test_gguf_repo_rejected_case_insensitively():
    with pytest.raises(RuntimeError):
        _check_model_format("Qwen/Qwen3-VL-2B-Instruct-gguf")


def test_mlx_repo_names_dropped_suffix_fix():
    with pytest.raises(RuntimeError) as exc:
        _check_model_format("mlx-community/Qwen2-VL-7B-Instruct-4bit-MLX")
    msg = str(exc.value)
    assert "MLX" in msg
    assert "mlx-community/Qwen2-VL-7B-Instruct-4bit" in msg


def test_awq_repo_rejected_when_autoawq_missing(monkeypatch):
    import app.model as model_mod

    monkeypatch.setattr(model_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError) as exc:
        _check_model_format("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    msg = str(exc.value)
    assert "AWQ" in msg
    assert "autoawq" in msg


def test_awq_repo_passes_when_autoawq_installed(monkeypatch):
    import app.model as model_mod

    monkeypatch.setattr(model_mod.importlib.util, "find_spec", lambda name: object())
    _check_model_format("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")


def test_gptq_repo_rejected_when_packages_missing(monkeypatch):
    import app.model as model_mod

    monkeypatch.setattr(model_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError) as exc:
        _check_model_format("Qwen/Qwen2.5-VL-7B-Instruct-GPTQ")
    msg = str(exc.value)
    assert "GPTQ" in msg
    assert "optimum" in msg or "gptqmodel" in msg


def test_gptq_repo_passes_when_packages_installed(monkeypatch):
    import app.model as model_mod

    monkeypatch.setattr(model_mod.importlib.util, "find_spec", lambda name: object())
    _check_model_format("Qwen/Qwen2.5-VL-7B-Instruct-GPTQ")
