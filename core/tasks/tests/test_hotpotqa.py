"""``core.tasks.hotpotqa``: answer extraction, official scoring, aggregation, records, tools and teams."""

import json

import pytest
from langchain_core.tools import tool

from core import teams, variant
from core.tasks import hotpotqa as task

WIKIPEDIA_TOOLS = ("wikipedia_search", "wikipedia_page")
TOOLLESS_WORKERS = {"query_decomposer_worker", "evidence_filter_worker", "citation_worker", "answer_simplifier_worker"}


def test_extract_answer_takes_the_last_answer_line():
    assert task.extract_answer("Answer: Paris.\nmore\n**Answer:** Lyon,") == "Lyon"
    assert task.extract_answer("Final answer: 1997") == "1997"
    assert task.extract_answer("no marker\n  last line  \n") == "last line"
    assert task.extract_answer("  \n") is None


def test_extract_manager_answer_ignores_terminate():
    assert task.extract_manager_answer("Answer: yes\nTERMINATE") == "yes"
    assert task.extract_answer("Paris\nTERMINATE") == "TERMINATE"
    assert task.extract_manager_answer("Paris\nTERMINATE") == "Paris"


def test_official_scoring():
    assert task.normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert task.exact_match_score("the Beatles", "Beatles") == 1.0
    assert task.f1_score("no", "yes") == (0.0, 0.0, 0.0)
    f1, precision, recall = task.f1_score("Paul McCartney", "James Paul McCartney")
    assert (precision, recall) == (1.0, 2 / 3) and f1 == pytest.approx(0.8)


def test_majority_over_normalized_answers():
    assert task.best_of_n(["Paris", None, "", "the Lyon", "lyon", "PARIS!", "Lyon"]) == "the Lyon"
    assert task.best_of_n(["Paris", "Lyon"]) == "Paris"
    assert task.best_of_n([None, ""]) is None
    answers = [{"answer": "Yes"}, {"answer": None}, {"answer": "yes."}, {"answer": "no"}]
    assert task.majority_vote(answers) == "Yes"
    assert task.vote_counts(answers) == {"yes": 2, "no": 1}


def test_records_round_scores_but_the_summary_sums_unrounded_f1():
    inst = {"id": "q1", "question": "Who?", "answer": "James Paul McCartney", "type": "bridge", "level": "hard"}
    rec = task.record(inst, "Paul McCartney", **task.meta(inst), latency_s=1.0)
    assert list(rec)[:8] == ["id", "question", "gold_answer", "predicted_answer", "em", "f1", "precision", "recall"]
    assert list(rec)[8:] == ["type", "level", "latency_s"]
    assert (rec["em"], rec["f1"], rec["recall"]) == (0.0, 0.8, 0.6667)
    missing = task.record({**inst, "id": "q2"}, None, latency_s=0.0)
    assert (missing["em"], missing["f1"], missing["precision"]) == (0.0, 0.0, 0.0)
    summary = task.summarize([rec, missing])
    assert (summary["n"], summary["n_extracted"]) == (2, 1)
    assert summary["f1"] == task.f1_score("Paul McCartney", "James Paul McCartney")[0] / 2
    assert " ~  em=0 f1=0.80 " in task.progress_line(0, 2, rec, [rec])


def test_wikipedia_tool_description_is_its_docstring(monkeypatch):
    search = tool(task.make_wikipedia_search(task.SEARCH_DOC))
    assert search.name == "wikipedia_search"
    assert search.description == task.SEARCH_DOC.rstrip()
    assert set(search.args) == {"query", "top_k"}
    page = task.make_wikipedia_page(task.PAGE_DOC_SHORT, options_label="options:")
    assert page.__name__ == "wikipedia_page" and page.__doc__ == task.PAGE_DOC_SHORT

    monkeypatch.setattr(
        task,
        "_wiki_request",
        lambda params: {
            "status": "ok",
            "http_status": 200,
            "data": {"query": {"pages": {"1": {"title": "Mercury", "pageprops": {"disambiguation": ""}}}}},
        },
    )
    assert json.loads(page("Mercury"))["status"] == "disambiguation"
    monkeypatch.setattr(task, "_wiki_request", lambda params: {"status": "ok", "http_status": 200, "data": {}})
    assert json.loads(search.invoke({"query": "nothing"}))["status"] == "no_results"


def test_wikipedia_reports_http_errors_and_bounded_content(monkeypatch):
    import requests

    seen = []

    class Response:
        status_code = 403

        def json(self):
            return {"query": {"pages": {"1": {"title": "Paris", "extract": "x" * 4001}}}}

    def get(url, **kwargs):
        seen.append((url, kwargs))
        return Response()

    monkeypatch.setattr(requests, "get", get)
    assert json.loads(task.read_page("Paris")) == {"status": "http_error", "http_status": 403}
    Response.status_code = 200
    result = json.loads(task.read_page("Paris"))
    assert result["status"] == "ok" and len(result["content"]) == 4000 and result["truncated"]
    assert seen[0][0].startswith("https://") and seen[0][1]["headers"]["User-Agent"]


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_hotpotqa_teams(size):
    assert len(teams.spec("sequential", task.DATASET, size).stages) == size
    centralized = teams.spec("centralized", task.DATASET, size)
    assert len(centralized.workers) == size - 1
    for worker in centralized.workers:
        assert worker.tools == (() if worker.role in TOOLLESS_WORKERS else WIKIPEDIA_TOOLS)
    assert teams.spec("independent", task.DATASET, size).recursion_limit == 20
    assert teams.spec("decentralized", task.DATASET, size).recursion_limit == 12


def test_team_size_variants_build_their_teams(monkeypatch):
    for name in ("INDEPENDENT_N_AGENTS", "HOTPOTQA_INDEPENDENT_RECURSION_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    centralized: dict = {"__name__": "hotpotqa_centralized_r8_under_test"}
    variant.load(centralized, "topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=8)
    assert centralized["_manager_system"]().endswith(
        "The seven workers are: retriever_worker, reasoner_worker, writer_worker, query_decomposer_worker, "
        "searcher_worker, evidence_filter_worker, citation_worker."
    )
    assert [t.name for t in centralized["MANAGER_TOOLS"]][:2] == list(WIKIPEDIA_TOOLS)
    assert len(centralized["DELEGATION_TOOLS"]) == 7 and centralized["MAX_TURNS"] == 18
    independent: dict = {"__name__": "hotpotqa_independent_r2_under_test"}
    variant.load(independent, "topologies.independent.hotpotqa.langgraph_hotpotqa", TEAM_SIZE=2)
    assert (independent["N_AGENTS"], independent["_RECURSION_LIMIT"]) == (2, 20)
