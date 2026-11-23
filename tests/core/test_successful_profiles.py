"""Successful manual settings are remembered without any probing or adaptive retries."""

import json

import httpx
import pytest

from core import checkpoints, profiles, providers, settings
from core.llm import chat_openai


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in (*profiles.FIELDS, "MAS_MODEL_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MAS_PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("MODEL_ID", "new-model")
    monkeypatch.setenv("VLLM_BASE_URL", "https://new-provider.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-to-be-saved")
    return tmp_path


def successful(fn=lambda: {"answer": "a wrong answer"}):
    with checkpoints.options(remember_model=True):
        return checkpoints.cached_call(fn)


def test_successful_unknown_model_is_saved_and_reused(isolated, monkeypatch):
    monkeypatch.setenv("TOKEN_LIMIT_FIELD", "max_completion_tokens")
    monkeypatch.setenv("MODEL_OMIT_PARAMS", "seed,temperature")
    monkeypatch.setenv("MODEL_EXTRA_BODY", '{"custom_option":true}')
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "1024")
    assert providers.provider(settings.base_url(), settings.model_id()) == "compatible"
    called = []
    output = successful(lambda: called.append(1) or {"answer": "wrong", "correct": False})
    assert output["correct"] is False and called == [1]
    path = next(profiles.root().glob("*.json"))
    data = json.loads(path.read_text())
    assert data["status"] == "successful" and data.get("verified_settings") is None
    assert data["successful_settings"]["TASK_MODEL_MAX_TOKENS"] == "1024"
    assert "not-to-be-saved" not in path.read_text() and "OPENAI_API_KEY" not in path.read_text()
    for key in ("TOKEN_LIMIT_FIELD", "MODEL_OMIT_PARAMS", "MODEL_EXTRA_BODY", "TASK_MODEL_MAX_TOKENS"):
        monkeypatch.delenv(key)
    decoding = settings.decoding()
    assert decoding["max_completion_tokens"] == 1024
    assert not {"temperature", "seed", "max_tokens"}.intersection(decoding)
    assert decoding["extra_body"] == {"custom_option": True}
    assert profiles.read(path.stem + "@successful") == data["successful_settings"]


def test_failed_trials_do_not_replace_success_or_trigger_retries(isolated, monkeypatch):
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "128")
    successful()
    path = next(profiles.root().glob("*.json"))
    original = json.loads(path.read_text())["successful_settings"]
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "999")
    profiles.remember()  # A fresh batch may record the trial before a request.
    called = []

    def fail():
        called.append(1)
        raise RuntimeError("unsupported parameter")

    with pytest.raises(RuntimeError, match="unsupported parameter"):
        successful(fail)
    assert called == [1]
    assert json.loads(path.read_text())["successful_settings"] == original
    monkeypatch.delenv("TASK_MODEL_MAX_TOKENS")
    assert settings.max_tokens() == 128
    successful(lambda: {"error": "BadRequestError: request rejected", "stage": "solve"})
    assert json.loads(path.read_text())["successful_settings"] == original
    successful(lambda: {"status": "execution_error", "failure_reason": "generation_failed"})
    assert json.loads(path.read_text())["successful_settings"] == original


def test_scoring_error_preserves_generation_config_and_resume_does_not_resave(isolated, monkeypatch):
    directory = isolated / "run.state"
    with checkpoints.options(remember_model=True), checkpoints.row_context(directory, {"id": "q"}):
        checkpoints.cached_call(lambda: {"answer": "generated"})
        with pytest.raises(ValueError, match="scoring failed"):
            raise ValueError("scoring failed")
    path = next(profiles.root().glob("*.json"))
    before = path.read_bytes()
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "500")
    with checkpoints.options(remember_model=True, resume=True), checkpoints.row_context(directory, {"id": "q"}):
        output = checkpoints.cached_call(lambda: pytest.fail("must reuse generation"))
    assert output["answer"] == "generated" and path.read_bytes() == before


def test_memory_is_separate_from_smoke_verification(isolated, monkeypatch):
    name = profiles.remember()
    profiles.main(["verify", name])
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "321")
    successful()
    assert profiles.read(name + "@successful")["TASK_MODEL_MAX_TOKENS"] == "321"
    assert profiles.read(name + "@verified")["TASK_MODEL_MAX_TOKENS"] == "32768"
    profiles.main(["save", name])
    assert profiles.read(name + "@successful")["TASK_MODEL_MAX_TOKENS"] == "321"


def test_connection_identity_and_explicit_preferences(isolated, monkeypatch):
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "123")
    successful()
    monkeypatch.delenv("TASK_MODEL_MAX_TOKENS")
    assert settings.max_tokens() == 123
    monkeypatch.setenv("MODEL_ID", "different")
    assert settings.max_tokens() == 32768
    monkeypatch.setenv("MODEL_ID", "new-model")
    monkeypatch.setenv("VLLM_BASE_URL", "https://different.example/v1")
    assert settings.max_tokens() == 32768
    monkeypatch.setenv("VLLM_BASE_URL", "https://new-provider.example/v1")
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "512")
    assert settings.max_tokens() == 512
    profiles.main(["save", "manual"])
    monkeypatch.delenv("TASK_MODEL_MAX_TOKENS")
    monkeypatch.setenv("MAS_MODEL_PROFILE", "manual")
    assert settings.max_tokens() == 512


def test_current_snapshot_is_saved_even_if_solver_changes_environment(isolated, monkeypatch):
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "64")

    def solve():
        monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "2048")
        return {"answer": "ok"}

    successful(solve)
    path = next(profiles.root().glob("*.json"))
    assert profiles.read(path.stem + "@successful")["TASK_MODEL_MAX_TOKENS"] == "64"


def test_profile_write_failure_does_not_discard_output(isolated, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise PermissionError("cannot write")

    monkeypatch.setattr(profiles, "remember", fail)
    assert successful() == {"answer": "a wrong answer"}
    assert "could not be saved" in caplog.text


def test_actual_sdk_makes_exactly_one_request_and_saves_success(isolated, monkeypatch):
    from openai import OpenAI

    monkeypatch.setenv("TOKEN_LIMIT_FIELD", "max_completion_tokens")
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "64")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "new-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "42"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    llm = chat_openai(max_retries=0)
    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    llm.client = OpenAI(api_key="test-only", base_url=settings.base_url(), http_client=http_client).chat.completions
    output = successful(lambda: {"answer": llm.invoke("What is 6*7?").content})
    assert output["answer"] == "42" and len(requests) == 1
    assert requests[0]["max_completion_tokens"] == 64
    assert len(list(profiles.root().glob("*.json"))) == 1
    http_client.close()
