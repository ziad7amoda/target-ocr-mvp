import pytest
from PIL import Image

from app.model import FakeEngine, GenerationRequest


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
