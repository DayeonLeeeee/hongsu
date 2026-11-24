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
        self.result = None   # 채점 응답을 바꾸려면 테스트에서 dict를 넣는다
        self.messages = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=lambda m: SimpleNamespace(id=m))

    def _create(self, **kw):
        self.calls.append(kw)
        if kw["tools"][0]["name"] == "refine_ocr_text":
            inp = {"corrected_text": "f(1) = \\frac{1}{2}", "changes": ["1/2"], "image_agreement": "matches", "note": ""}
        else:
            inp = self.result or good()
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
    from hongsu import guard
    guard.reset()   # 요청 횟수 카운터는 테스트마다 새로
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


# ── 통합: /grade 전체 흐름 ──────────────────────────────────


def _saved(monkeypatch):
    from hongsu import storage
    rows = []
    monkeypatch.setattr(storage, "save_rv2", lambda r: rows.append(r))
    return rows


def test_grade_from_ocr_steps_numbers_lines_and_saves(env, monkeypatch):
    """학생 화면 경로: OCR로 확인한 단계(solution_steps) → 줄 번호 붙은 답안으로 채점·저장."""
    rows = _saved(monkeypatch)
    steps = [{"index": 1, "text": "f(1)=a"}, {"index": 2, "text": "f^(-1)(a)=2"}]
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_steps": steps,
                                        "student_id": "s1", "attempt_count": 2}).get_json()
    assert j["success"] and j["outcome"] == "graded"
    assert "1) f(1)=a\n2) f^(-1)(a)=2" in env.claude.calls[0]["messages"][0]["content"]
    assert len(rows) == 1 and rows[0]["answer_text"] == "1) f(1)=a\n2) f^(-1)(a)=2"
    assert rows[0]["student_id"] == "s1" and rows[0]["raw_responses"] and rows[0]["reveal_answer"] is False


def test_grade_image_only_is_rejected_without_llm(env):
    r = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_image_b64": "QUJD"})
    assert r.status_code == 400 and env.claude.calls == []


def test_grade_abstain_goes_to_review_and_reveal_after_3_wrong(env, monkeypatch):
    rows = _saved(monkeypatch)
    res = good()
    res.update(main_error_code=None, review_needed=True, review_reason="기준표 밖 오류")
    env.claude.result = res
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER,
                                        "attempt_count": 4}).get_json()
    assert j["outcome"] == "review" and j["review_sources"] == ["model"] and j["review_needed"]
    assert j["reveal_answer"] is True and rows[0]["outcome"] == "review"


def test_grade_incomplete_answer(env):
    res = good()
    res.update(response_complete=False, missing_criteria=["5"])
    env.claude.result = res
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER}).get_json()
    assert j["outcome"] == "incomplete" and j["missing_criteria"] == [5]


def test_grade_api_failure_still_answers_with_error(env, monkeypatch):
    def boom(**kw):
        raise RuntimeError("overloaded")
    monkeypatch.setattr(env.claude.messages, "create", boom)
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER}).get_json()
    assert j["success"] is False and j["outcome"] == "error" and "overloaded" in j["error"]


# ── 통합: /admin/problems ──────────────────────────────────


def test_problem_list_filter_and_bulk(env):
    c = env.client
    added = c.post("/admin/problems/bulk", json={"problems": [{"title": "a", "unit": "유리함수"},
                                                             {"id": "Z1", "title": "b", "unit": "유리함수"}]})
    ids = added.get_json()["added_ids"]
    assert ids[1] == "Z1" and len(ids) == 2
    j = c.get("/admin/problems?unit=유리함수").get_json()
    assert {p["id"] for p in j["problems"]} >= set(ids) and all(p["unit"] == "유리함수" for p in j["problems"])
    assert any(p["id"] == "INV-R-01" for p in c.get("/admin/problems").get_json()["problems"])
    for pid in ids:
        c.delete(f"/admin/problems/{pid}")


def test_problem_unknown_id_is_404(env):
    c = env.client
    assert c.get("/admin/problems/NOPE").status_code == 404
    assert c.put("/admin/problems/NOPE", json={"title": "x"}).status_code == 404
    assert c.delete("/admin/problems/NOPE").status_code == 404


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="실제 PostgreSQL이 있을 때만 (TEST_DATABASE_URL)")
def test_postgres_real_roundtrip(env, monkeypatch):
    """빈 DB에 테이블을 만들고, /grade 결과가 jsonb·한글까지 그대로 들어가는지."""
    import psycopg
    import hongsu.db as db
    url = os.environ["TEST_DATABASE_URL"]
    with psycopg.connect(url) as conn:
        conn.execute("DROP TABLE IF EXISTS rv2_gradings")
    monkeypatch.setattr(db, "DATABASE_URL", url)
    monkeypatch.setattr(db, "_ready", False)
    text = ANSWER + "\n3) 답 5 (f⁻¹ 확인)"
    env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": text})
    assert db.check() == {"ok": True, "backend": "postgres", "rows": 1}
    with psycopg.connect(url) as conn:
        row = conn.execute("SELECT answer_text, lines, outcome, reveal_answer, raw_responses IS NOT NULL "
                           "FROM rv2_gradings").fetchone()
    assert row[0] == text and row[1][1]["code"] == "R8" and row[2] == "graded"
    assert row[3] is False and row[4] is True


# ── 입력 검증·요청 횟수 제한·재현성 (Day 5) ───────────────────


@pytest.mark.parametrize("path,body", [
    ("/grade", None),
    ("/grade", {"problem_id": "INV-R-01", "solution_text": ANSWER, "attempt_count": "많이"}),
    ("/grade", {"problem_id": "INV-R-01", "solution_text": ANSWER, "attempt_count": 0}),
    ("/grade", {"problem_id": "INV-R-01", "solution_text": "가" * 5001}),
    ("/grade", {"problem_id": "INV-R-01", "solution_steps": "1) f(1)=3"}),
    ("/grade", {"problem_id": "INV-R-01", "solution_steps": [{"text": "x"}] * 61}),
    ("/grade", {"problem_id": "INV-R-01", "solution_text": ANSWER, "input_status": "blank"}),
    ("/grade", {"problem_id": ["INV-R-01"], "solution_text": ANSWER}),
    ("/ocr", {"mime_type": "image/png"}),
    ("/refine", {"image": "QUJD", "instruction": "x" * 1001}),
    ("/measure_consistency", {"problem_id": "INV-R-01", "solution_text": ANSWER, "n_runs": "x"}),
    ("/admin/problems", None),
    ("/admin/problems/bulk", {"problems": "A"}),
    ("/results", None),
    ("/master_solution", None),
])
def test_bad_input_is_400_not_500_and_costs_nothing(env, path, body):
    r = env.client.post(path, json=body) if body is not None else env.client.post(path, data="x")
    assert r.status_code == 400, (path, r.status_code, r.get_json())
    assert r.get_json()["success"] is False and r.get_json()["error"]
    assert env.claude.calls == []


def test_numbers_as_strings_still_work(env):
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER,
                                        "attempt_count": "2", "student_id": 1234}).get_json()
    assert j["success"] and j["attempt_count"] == 2 and j["student_id"] == "1234"


def test_too_large_body_is_413(env, monkeypatch):
    env.app.config["MAX_CONTENT_LENGTH"], old = 1000, env.app.config["MAX_CONTENT_LENGTH"]
    try:
        assert env.client.post("/ocr", json={"image": "A" * 2000}).status_code == 413
    finally:
        env.app.config["MAX_CONTENT_LENGTH"] = old


def test_rate_limit_per_student_and_ip(env, monkeypatch):
    from hongsu import guard
    monkeypatch.setattr(guard, "RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(guard, "RATE_LIMIT_IP_PER_MIN", 3)
    body = {"problem_id": "INV-R-01", "solution_text": ANSWER, "student_id": "s1"}
    assert [env.client.post("/grade", json=body).status_code for _ in range(3)] == [200, 200, 429]
    r = env.client.post("/grade", json=body)
    assert int(r.headers["Retry-After"]) > 0 and "학생별" in r.get_json()["error"]
    other = dict(body, student_id="s2")    # 다른 학생: 학생 한도는 따로, 같은 교실 IP 한도는 공유
    assert env.client.post("/grade", json=other).status_code == 200
    assert env.client.post("/grade", json=other).status_code == 429   # IP 3회 초과
    assert "IP별" in env.client.post("/grade", json=other).get_json()["error"]
    # 다른 IP(X-Forwarded-For)의 학생은 영향 없음, 다른 엔드포인트도 따로 센다
    assert env.client.post("/grade", json=dict(body, student_id="s3"),
                           headers={"X-Forwarded-For": "10.0.0.9"}).status_code == 200
    assert env.client.post("/refine", json={"image": "QUJD", "current_text": "a", "instruction": "b",
                                            "student_id": "s1"}).status_code == 200
    assert len([c for c in env.claude.calls if c["tools"][0]["name"] != "refine_ocr_text"]) == 4


def test_rate_limit_skips_internal_consistency_calls(env, monkeypatch):
    from hongsu import guard
    monkeypatch.setattr(guard, "RATE_LIMIT_PER_MIN", 1)
    monkeypatch.setattr(guard, "RATE_LIMIT_IP_PER_MIN", 1)
    r = env.client.post("/measure_consistency", json={"problem_id": "INV-R-01", "solution_text": ANSWER,
                                                      "n_runs": 3, "student_id": "s1"})
    assert r.status_code == 200 and len(r.get_json()["runs"]) == 3 and all(x["success"] for x in r.get_json()["runs"])


def test_rate_limit_off_with_zero(env, monkeypatch):
    from hongsu import guard
    monkeypatch.setattr(guard, "RATE_LIMIT_PER_MIN", 0)
    monkeypatch.setattr(guard, "RATE_LIMIT_IP_PER_MIN", 0)
    body = {"problem_id": "INV-R-01", "solution_text": ""}
    assert all(env.client.post("/grade", json=body).status_code == 200 for _ in range(5))


def test_git_commit_recorded_in_result_health_and_saved_row(env, monkeypatch):
    import rv2_grader
    monkeypatch.setattr(rv2_grader, "_COMMIT", "abc123def456")
    monkeypatch.setattr(rv2_grader, "_COMMIT_DONE", True)
    rows = _saved(monkeypatch)
    j = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ANSWER}).get_json()
    assert j["git_commit"] == "abc123def456" and rows[0]["git_commit"] == "abc123def456"
    blank = env.client.post("/grade", json={"problem_id": "INV-R-01", "solution_text": ""}).get_json()
    assert blank["git_commit"] == "abc123def456"
    assert env.client.get("/health").get_json()["git_commit"] == "abc123def456"


def test_code_commit_prefers_deploy_env(monkeypatch):
    import rv2_grader
    monkeypatch.setattr(rv2_grader, "_COMMIT", None)       # 테스트 뒤 원래 값으로 돌아가게
    monkeypatch.setattr(rv2_grader, "_COMMIT_DONE", False)
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "0123456789abcdef0123")
    assert rv2_grader.code_commit() == "0123456789ab"
