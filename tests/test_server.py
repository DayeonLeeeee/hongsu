# coding: utf-8
"""서버 엔드포인트 테스트 (외부 API 없음: Claude·Gemini·Supabase는 가짜로 바꾼다).

실행: 저장소 루트에서  python -m pytest tests -q
"""
import base64
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ANTHROPIC_API_KEY", "test")

from tests.test_rv2_sanitize import _tool_block, good  # noqa: E402

ANSWER = "1) f(1)=a\n2) f^(-1)(a)=2"

# 분리 전 hongsu_server.py의 URL 목록. 구조를 바꿔도 이 목록은 그대로여야 한다.
EXPECTED_ROUTES = {
    "POST /ocr", "POST /refine", "POST /grade", "POST /measure_consistency",
    "GET /admin/problems", "POST /admin/problems", "GET /admin/problems/<pid>",
    "PUT /admin/problems/<pid>", "DELETE /admin/problems/<pid>", "POST /admin/problems/bulk",
    "POST /admin/rv2/reload", "POST /master_solution",
    "POST /results", "GET /results", "GET /results/<rid>", "GET /stats/error_distribution",
    "GET /results/export", "GET /health", "GET /admin", "GET /student", "GET /manifest.json",
}


class FakeClaude:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=lambda m: SimpleNamespace(id=m))

    def _create(self, **kw):
        self.calls.append(kw)
        if kw["tools"][0]["name"] == "refine_ocr_text":
            inp = {"corrected_text": "f(1) = \\frac{1}{2}", "changes": ["1/2"], "image_agreement": "matches", "note": ""}
        else:
            inp = good()
        return SimpleNamespace(content=[_tool_block(inp)], stop_reason="end_turn", model=kw["model"],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                                     cache_creation_input_tokens=0, cache_read_input_tokens=0))


class FakeGemini:
    def generate_content(self, parts, generation_config=None):
        return SimpleNamespace(text=json.dumps({"steps": [{"index": 1, "text": "f(x)=\\sqrt{x}"}]}))


@pytest.fixture()
def env(monkeypatch):
    import hongsu.clients as clients
    import hongsu_server
    fake = FakeClaude()
    monkeypatch.setattr(clients, "get_claude", lambda: fake)
    monkeypatch.setattr(clients, "get_supabase", lambda: None)
    monkeypatch.setattr(clients, "get_gemini_model", lambda: FakeGemini())
    return SimpleNamespace(client=hongsu_server.app.test_client(), app=hongsu_server.app, claude=fake)


def test_routes_unchanged(env):
    got = {f"{m} {r.rule}" for r in env.app.url_map.iter_rules() if r.endpoint != "static"
           for m in r.methods if m not in ("HEAD", "OPTIONS")}
    assert got == EXPECTED_ROUTES


def test_grade_returns_outcome_and_request_id(env):
    r = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER},
                        headers={"X-Request-ID": "t-1"})
    j = r.get_json()
    assert r.status_code == 200 and j["success"]
    assert j["outcome"] == "graded" and j["main_error_code"] == "R8"
    assert j["request_id"] == "t-1" and r.headers["X-Request-ID"] == "t-1"
    assert j["model"] == j["response_models"][0]


def test_grade_blank_answer_skips_llm(env):
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ""}).get_json()
    assert j["outcome"] == "not_attempted" and env.claude.calls == []


def test_grade_rejects_unknown_problem_and_bad_mode(env):
    assert env.client.post("/grade", json={"problem_id": "NOPE", "solution_text": ANSWER}).status_code == 400
    assert env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER,
                                           "mode": "bad"}).status_code == 400


def test_measure_consistency_keeps_outer_request_id(env):
    """내부에서 /grade를 다시 부를 때 바깥 요청이 죽던 버그(1.3 점검에서 발견)의 회귀 방지."""
    r = env.client.post("/measure_consistency", json={"problem_id": "INV-R-01", "solution_text": ANSWER,
                                                      "n_runs": 2}, headers={"X-Request-ID": "outer"})
    j = r.get_json()
    assert r.status_code == 200 and j["code_consistency"] == 1.0
    assert r.headers["X-Request-ID"] == "outer"
    assert all(run["request_id"] != "outer" for run in j["runs"])


def test_ocr_and_refine_normalize_latex(env):
    img = base64.b64encode(b"img").decode()
    j = env.client.post("/ocr", json={"image": img}).get_json()
    assert j["steps"] == [{"index": 1, "text": "f(x)=√x"}]
    j = env.client.post("/refine", json={"image": img, "current_text": "f(1)", "instruction": "x"}).get_json()
    assert j["success"] and j["corrected_text"] == "f(1) = (1/2)"


def test_problem_crud_and_reload(env):
    c = env.client
    pid = c.post("/admin/problems", json={"title": "t"}).get_json()["id"]
    assert c.put(f"/admin/problems/{pid}", json={"title": "t2", "id": "x"}).get_json()["problem"]["id"] == pid
    assert c.delete(f"/admin/problems/{pid}").get_json()["success"]
    assert c.get(f"/admin/problems/{pid}").status_code == 404
    assert c.post("/admin/rv2/reload").get_json()["problems"] >= 1
    assert c.get("/admin/problems/INV-R-01").status_code == 200


def test_results_stats_and_export(env):
    c = env.client
    c.post("/results", json={"problem_id": "INV-R-01", "is_correct": False, "main_error_code": "R8"})
    assert c.get("/stats/error_distribution").get_json()["distribution"].get("R8", 0) >= 1
    r = c.get("/results/export")
    assert r.status_code == 200 and r.data.decode("utf-8").startswith("result_id,")


def test_health_and_pages(env):
    c = env.client
    assert c.get("/health").get_json()["status"] == "ok"
    j = c.get("/health?deep=1").get_json()
    assert j["checks"]["claude"]["ok"] is True and j["checks"]["supabase"]["ok"] is False
    for path in ("/student", "/admin", "/manifest.json"):
        assert c.get(path).status_code == 200


def test_postgres_storage_when_database_url_set(env, monkeypatch):
    """DATABASE_URL이 있으면 Postgres에 저장한다. DB에 못 붙어도 채점 응답은 나가고 /health가 알려 준다."""
    import hongsu.db as db
    monkeypatch.setattr(db, "DATABASE_URL", "postgresql://u:p@127.0.0.1:1/x")
    monkeypatch.setattr(db, "_ready", False)
    saved = []
    monkeypatch.setattr(db, "insert_rv2", lambda row: saved.append(row))
    r = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER})
    assert r.get_json()["success"] and saved and saved[0]["problem_id"] == "INV-R-01"
    assert "raw_responses" in saved[0] and "outcome" in saved[0]
    assert "reveal_answer" in saved[0] and "feedback" in saved[0]
    monkeypatch.undo()
    monkeypatch.setattr(db, "DATABASE_URL", "postgresql://u:p@127.0.0.1:1/x")
    monkeypatch.setattr(db, "_ready", False)
    import hongsu.clients as clients
    monkeypatch.setattr(clients, "get_claude", lambda: env.claude)
    r = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER})
    assert r.get_json()["success"]          # 저장 실패해도 채점은 응답
    j = env.client.get("/health?deep=1").get_json()
    assert j["checks"]["db"]["ok"] is False and "supabase" not in j["checks"]
