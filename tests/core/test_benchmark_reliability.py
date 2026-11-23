"""Offline regressions for provider requests, durable resume and tool-call boundaries."""

import json
import os

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from core import batch, checkpoints, profiles, providers
from core.llm import chat_openai
from core.runtime import closed_tool_history, safe_tool_node


@pytest.fixture(autouse=True)
def isolate_model(monkeypatch):
    for key in (*profiles.FIELDS, "MAS_MODEL_PROFILE"):
        monkeypatch.delenv(key, raising=False)


def test_qwen_actual_sdk_request_preserves_cap_and_thinking(monkeypatch):
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "64")
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5-27b",
                "choices": [
                    {"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "answer"}}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 64, "total_tokens": 68},
            },
        )

    llm = chat_openai(model="qwen3.5-27b", base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1")
    client = httpx.Client(transport=httpx.MockTransport(handle))
    from openai import OpenAI

    llm.client = OpenAI(api_key="test-only", base_url=llm.openai_api_base, http_client=client).chat.completions
    result = llm.invoke("hello")
    assert bodies[0]["max_tokens"] == 64 and "max_completion_tokens" not in bodies[0]
    assert bodies[0]["enable_thinking"] is False
    assert "chat_template_kwargs" not in bodies[0]
    assert result.usage_metadata["output_tokens"] == 64
    assert result.response_metadata["finish_reason"] == "length"


def test_provider_specific_fields_and_overrides(monkeypatch):
    assert providers.body("https://api.deepseek.com/v1", "deepseek-chat", False, 1.05) == {
        "thinking": {"type": "disabled"}
    }
    assert providers.body("https://api.openai.com/v1", "gpt", False, 1.05) == {}
    body = providers.adapt_payload({"max_tokens": 123, "tool_choice": "none"}, "https://api.openai.com/v1", "gpt")
    assert body == {"max_completion_tokens": 123}
    monkeypatch.setenv("MODEL_OMIT_PARAMS", "temperature,top_p,seed")
    assert (
        providers.adapt_payload({"temperature": 0, "top_p": 0.9, "seed": 0}, "https://api.openai.com/v1", "gpt") == {}
    )


def test_profiles_preserve_verified_revision_and_exclude_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_ID", "qwen-test")
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-be-saved")
    profiles.main(["save", "qwen"])
    profiles.main(["verify", "qwen"])
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "123")
    profiles.main(["save", "qwen"])
    saved = (tmp_path / "qwen.json").read_text()
    assert "must-never-be-saved" not in saved and "OPENAI_API_KEY" not in saved
    assert profiles.read("qwen")["TASK_MODEL_MAX_TOKENS"] == "123"
    assert profiles.read("qwen@verified")["TASK_MODEL_MAX_TOKENS"] == "32768"
    monkeypatch.delenv("MODEL_ID")
    monkeypatch.setenv("MAS_MODEL_PROFILE", "qwen@verified")
    from core import settings

    assert settings.model_id() == "qwen-test"


def test_automatic_model_memory_keeps_latest_separate_from_verified(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_PROFILE_DIR", str(tmp_path))
    name = profiles.remember()
    profiles.main(["verify", name])
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "42")
    assert profiles.remember() == name
    assert profiles.read(name)["TASK_MODEL_MAX_TOKENS"] == "42"
    assert profiles.read(name + "@verified")["TASK_MODEL_MAX_TOKENS"] == "32768"
    monkeypatch.setenv("MODEL_ID", "another-model")
    assert profiles.remember() != name


def test_scoring_crash_resumes_saved_generation_without_calling_model(tmp_path):
    path = tmp_path / "results.jsonl"
    called = []

    def row(_, instance):
        output, _, error = batch.attempt(
            lambda: called.append(instance["id"]) or {"answer": "full answer", "telemetry": {"total_tokens": 7}}
        )
        assert error is None
        if len(called) == 1 and not scoring_fixed[0]:
            raise RuntimeError("scorer crashed")
        return {"id": instance["id"], "correct": False, **output["telemetry"]}

    instances = [{"id": "a"}, {"id": "b"}]
    scoring_fixed = [False]
    with pytest.raises(RuntimeError, match="scorer crashed"):
        batch.run_batch(instances, row, summarize=lambda rows: {}, out_path=path)
    artifact = list((tmp_path / "results.jsonl.state/artifacts").glob("*.json"))[0]
    assert json.loads(artifact.read_text())["output"]["answer"] == "full answer"
    scoring_fixed[0] = True
    result = batch.run_batch(instances, row, summarize=lambda rows: {}, out_path=path, resume=True)
    assert called == ["a", "b"]
    assert len(result["per_instance"]) == 2
    batch.run_batch(
        instances,
        lambda *_: pytest.fail("completed answer must not rerun"),
        summarize=lambda rows: {},
        out_path=path,
        resume=True,
        retry_execution_errors=True,
    )


def test_retry_only_execution_faults_and_reject_changed_run(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    rows = [{"id": "wrong"}, {"id": "network"}]
    batch.run_batch(
        rows,
        lambda _, r: (
            {"id": r["id"], "correct": False, "error": "API result mismatch", "stage": "score"}
            if r["id"] == "wrong"
            else {"id": r["id"], "error": "APIConnectionError: offline", "stage": "solve"}
        ),
        summarize=lambda rows: {},
        out_path=path,
    )
    calls = []
    batch.run_batch(
        rows,
        lambda _, r: calls.append(r["id"]) or {"id": r["id"], "correct": True},
        summarize=lambda rows: {},
        out_path=path,
        resume=True,
        retry_execution_errors=True,
    )
    assert calls == ["network"]
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "99")
    with pytest.raises(ValueError, match="resume_config_mismatch"):
        batch.run_batch(rows, lambda *_: {}, summarize=lambda rows: {}, out_path=path, resume=True)
    assert len(path.read_text().splitlines()) == 2


def test_completed_rows_survive_truncated_jsonl_view(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [{"id": "a"}, {"id": "b"}]
    batch.run_batch(rows, lambda _, r: {**r, "correct": True}, summarize=lambda rows: {}, out_path=path)
    path.write_text("")  # simulates interruption while rebuilding the exported view
    result = batch.run_batch(
        rows,
        lambda *_: pytest.fail("durable records should be reused"),
        summarize=lambda rows: {},
        out_path=path,
        resume=True,
    )
    assert len(result["per_instance"]) == 2


def test_tool_history_closes_calls_without_executing_or_losing_usage():
    ai = AIMessage(
        content="",
        id="ai1",
        tool_calls=[{"name": "missing", "args": {}, "id": "call1"}],
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    )
    history = closed_tool_history([HumanMessage("q"), ai])
    assert history[1].usage_metadata == ai.usage_metadata
    assert history[1].id == "ai1"
    assert isinstance(history[2], ToolMessage) and history[2].tool_call_id == "call1"
    assert history[2].status == "error"
    assert len(closed_tool_history(history)) == 3


def test_safe_tool_node_handles_malformed_arguments_and_runs_valid_calls():
    executed = []

    @tool
    def double(value: int) -> int:
        """Double a number."""
        executed.append(value)
        return 2 * value

    graph = StateGraph(MessagesState)
    graph.add_node("tools", safe_tool_node([double]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "double", "args": {"value": 3}, "id": "good"}],
        invalid_tool_calls=[{"name": "double", "args": "broken-json", "id": "bad", "error": "invalid"}],
    )
    messages = graph.compile().invoke({"messages": [ai]})["messages"]
    replies = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    assert executed == [3]
    assert replies["good"].content == "6" and replies["bad"].status == "error"


@pytest.mark.skipif(os.name == "nt", reason="fork workers require WSL/Linux")
def test_two_workers_isolate_instances_and_write_ordered_results(tmp_path):
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def row(_, instance):
        output = checkpoints.cached_call(lambda: {"answer": instance["id"], "pid": os.getpid()})
        return {"id": output["answer"], "pid": output["pid"], "correct": True}

    result = batch.run_batch(rows, row, summarize=lambda rows: {}, out_path=tmp_path / "results.jsonl", workers=2)
    assert [r["id"] for r in result["per_instance"]] == ["a", "b", "c"]
    assert all(r["pid"] != os.getpid() for r in result["per_instance"])
    assert len(list((tmp_path / "results.jsonl.state/artifacts").glob("*.json"))) == 3


def test_cli_protection_keeps_first_result_and_failed_output_is_not_reused(tmp_path):
    path = tmp_path / "results.jsonl"
    with checkpoints.options(protect_outputs=True):
        batch.run_batch([{"id": "a"}], lambda _, r: {**r, "correct": True}, summarize=lambda rows: {}, out_path=path)
        first = path.read_bytes()
        with pytest.raises(FileExistsError, match="checkpoint_exists"):
            batch.run_batch(
                [{"id": "a"}], lambda *_: pytest.fail("must not execute"), summarize=lambda rows: {}, out_path=path
            )
        assert path.read_bytes() == first
    with checkpoints.row_context(tmp_path / "failed", {"id": "a"}):
        checkpoints.cached_call(lambda: {"error": "APIConnectionError: offline", "stage": "solve"})
    with checkpoints.options(resume=True), checkpoints.row_context(tmp_path / "failed", {"id": "a"}):
        assert checkpoints.cached_call(lambda: {"answer": "recovered"}) == {"answer": "recovered"}


def test_toolhop_finalization_respects_profile_and_endpoint_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletion

    from core.tasks import toolhop
    from core.telemetry import normalize

    monkeypatch.setenv("MAS_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("TASK_MODEL_MAX_TOKENS", "64")
    profiles.main(["save", "test"])
    monkeypatch.delenv("TASK_MODEL_MAX_TOKENS")
    monkeypatch.setenv("MAS_MODEL_PROFILE", "test")
    monkeypatch.setenv("MODEL_OMIT_PARAMS", "temperature,top_p,seed")
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<answer>1</answer>"},
                    }
                ],
            }
        )

    llm = SimpleNamespace(
        base_url="https://api.openai.com/v1", chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    toolhop._force_final_answer(llm, [], normalize(None), 0, model="gpt")
    assert sent[0]["max_completion_tokens"] == 64
    assert not {"max_tokens", "temperature", "top_p", "seed"}.intersection(sent[0])
