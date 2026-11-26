# coding: utf-8
"""단위 테스트: 채점 검증(validate), 유틸, 코드북 조회, 프롬프트 고정, 회귀 테스트 비교 (LLM 호출 없음).

실행: 저장소 루트에서  python -m pytest tests -q
"""
import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import rv2_codebook  # noqa: E402
import rv2_grader as g  # noqa: E402
from hongsu import utils  # noqa: E402
from prompts import rv2 as p_rv2  # noqa: E402
from tests.test_rv2_sanitize import ANSWER, PID, good  # noqa: E402

# ── 채점 검증 validate(): 키트 run_eval.validate()와 같은 기계 검증 ───────────────


def test_validate_passes_clean_result():
    assert g.validate(good(), ANSWER, ["R8"], True, [1, 2, 3, 4, 5]) == []


def test_validate_rejects_quote_not_in_answer():
    """환각 차단: wrong 줄의 인용이 학생 답안에 없으면 검수 사유."""
    p = good()
    p["lines"][1]["quote"] = "f(3)=7"
    assert any("quote가 답안 원문에 없음" in i for i in g.validate(p, ANSWER, ["R8"], True))


def test_validate_quote_ignores_whitespace_only():
    p = good()
    p["lines"][1]["quote"] = "f^(-1) (a) = 2"
    assert g.validate(p, ANSWER, ["R8"], True) == []


def test_validate_wrong_line_needs_quote():
    p = good()
    p["lines"][1]["quote"] = None
    assert any("quote 없음" in i for i in g.validate(p, ANSWER, ["R8"], True))


def test_validate_rejects_code_outside_problem():
    p = good()
    p["lines"][1]["code"] = "R5"
    p["main_error_code"] = "R5"
    issues = g.validate(p, ANSWER, ["R8"], True)
    assert any("허용 밖 코드 R5" in i for i in issues) and any("main_error_code 허용 밖" in i for i in issues)


def test_validate_grade_only_skips_code_check():
    p = good()
    p["lines"][1]["code"] = "R5"
    p["main_error_code"] = "R5"
    assert g.validate(p, ANSWER, [], False) == []


def test_validate_missing_criteria_must_be_criterion_numbers():
    p = good()
    p["missing_criteria"] = [3, 9]
    issues = g.validate(p, ANSWER, ["R8"], True, [1, 2, 3, 4, 5])
    assert len(issues) == 1 and "[9]" in issues[0]


# ── 유틸 ──────────────────────────────────────────────────


@pytest.mark.parametrize("v,want", [
    (5, 5), ("5", 5), ("⑤", 5), ("기준5", 5), (3.0, 3), ("3.0", 3),
    (3.5, None), ("2.5", None), (True, None), (None, None), ("", None), ("없음", None),
])
def test_to_int(v, want):
    assert g._to_int(v) == want


def test_steps_to_answer_text_numbers_lines_and_skips_blank():
    steps = [{"index": 1, "text": " f(1)=3 "}, {"index": 2, "text": ""}, {"index": 3, "text": "답 5"}]
    assert g.steps_to_answer_text(steps) == "1) f(1)=3\n3) 답 5"
    assert g.steps_to_answer_text([{"text": "a"}, {"text": "b"}]) == "1) a\n2) b"


def test_to_server_steps_keeps_front_fields():
    p = good()
    p["lines"][1]["depends_on"] = [1, 2]
    steps = g.to_server_steps(p)
    assert [s["index"] for s in steps] == [1, 2]
    assert steps[1]["status"] == "wrong" and steps[1]["code"] == "R8" and steps[1]["depends_on_line_id"] == 1
    assert g.to_server_steps(None) == []


def test_normalize_plain_latex():
    assert utils.normalize_plain(r"$f^{-1}(x)=\frac{x-2}{3}$") == "f^-1(x)=((x-2)/3)"   # x−2/3이 되면 안 됨
    assert utils.normalize_plain(r"\frac{1}{2} + \frac{a+b}{c-1}") == "(1/2) + ((a+b)/(c-1))"
    assert utils.normalize_plain(r"\sqrt{x} \leq 2 \times \pi") == "√x ≤ 2 × π"
    assert utils.normalize_plain(r"\sqrt{x+1}") == "√(x+1)"   # √x+1이 되면 안 됨
    assert utils.normalize_plain("") == ""


def test_parse_tool_use_reads_tool_or_text_and_raises_on_nothing():
    from types import SimpleNamespace as NS
    assert utils.parse_tool_use(NS(content=[NS(type="tool_use", input={"a": 1})])) == {"a": 1}
    assert utils.parse_tool_use(NS(content=[NS(type="text", text='결과: {"a": 2}')])) == {"a": 2}
    with pytest.raises(ValueError):
        utils.parse_tool_use(NS(content=[NS(type="text", text="없음")]))


def test_make_image_block():
    b = utils.make_image_block("QUJD", "image/png")
    assert b == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}


# ── 코드북·프롬프트 ───────────────────────────────────────


def test_codebook_lookups():
    assert rv2_codebook.has_problem(PID) and not rv2_codebook.has_problem("NOPE")
    assert rv2_codebook.allowed_codes(PID) == ["R8"]
    assert [int(r["line_no"]) for r in rv2_codebook.lines_for(PID)] == [1, 2, 3, 4, 5]
    with pytest.raises(KeyError):
        rv2_codebook.problem("NOPE")


def test_every_problem_builds_a_prompt_with_its_codes():
    mats = rv2_codebook.get()
    for pr in mats["problems"]:
        b = p_rv2.build_prompt(mats, pr["problem_id"])
        assert "{STUDENT_ANSWER}" in b["user_template"]
        assert b["allowed_codes"] and all(c in b["system"] + b["user_template"] for c in b["allowed_codes"])


# 프롬프트(시스템·사용자 틀·도구 스키마) 지문. 프롬프트를 바꾸면 이 테스트가 실패한다.
# → prompts/rv2.py의 PORT_VERSION(또는 엑셀)을 올리고, 실패 메시지의 새 지문을 여기에 적는다.
#   버전을 안 올리고 프롬프트만 바뀌면 결과 파일의 prompt_version으로 두 프롬프트를 구분할 수 없게 된다.
PROMPT_FINGERPRINT = {
    "R-v3-with_codebook-kit-1.2-20261022-hongsu-rv2-1.3-xdfcadb76": "a843fc32e2382099",
    "R-v3-with_codebook-kit-1.2-20261022-hongsu-rv2-1.4-xdfcadb76": "7a1d2b54929654b9",  # 1.4: error_criterion 칸
    "R-v3-with_codebook-kit-1.2-20261022-hongsu-rv2-1.5-xdfcadb76": "877183f70c90d0ec",  # 1.5: stopped_midway 칸
    "R-v3-with_codebook-kit-1.2-20261022-hongsu-rv2-1.6-xdfcadb76": "edf92c57f143a7fb",  # 1.6: 1.3 + violated_criterion (stopped_midway 없음)
    "R-v3-with_codebook-kit-1.2-20261022-hongsu-rv2-1.7-xdfcadb76": "edf92c57f143a7fb",  # 1.7: dataset_rv2로는 1.6과 같은 문구 (R-v3.1 자료 지원만 추가)
}


def _prompt_fingerprint():
    mats = rv2_codebook.get()
    h = hashlib.sha256()
    for pr in mats["problems"]:
        for mode in ("with_codebook", "grade_only"):
            b = p_rv2.build_prompt(mats, pr["problem_id"], mode)
            schema = json.dumps(p_rv2.tool_schema(mode), ensure_ascii=False, sort_keys=True)
            h.update((b["system"] + "\x00" + b["user_template"] + "\x00" + schema).encode())
    return p_rv2.build_prompt(mats, PID)["version"], h.hexdigest()[:16]


def test_prompt_change_requires_version_bump():
    version, fp = _prompt_fingerprint()
    assert version in PROMPT_FINGERPRINT, (
        f"새 프롬프트 버전 {version}: tests/test_units.py PROMPT_FINGERPRINT에 \"{version}\": \"{fp}\" 추가")
    assert PROMPT_FINGERPRINT[version] == fp, (
        f"프롬프트 내용이 바뀌었는데 버전({version})은 그대로다. prompts/rv2.py PORT_VERSION을 올릴 것 (새 지문 {fp})")


# ── 회귀 테스트(실험 3-3) 비교 ─────────────────────────────


def _rec(aid, code="R8", first=2, outcome="graded"):
    lines = [{"no": 1, "judgment": "wrong", "matched_criterion": first}] if first else []
    return {"answer_id": aid, "outcome": outcome, "prompt_version": "v", "grader_version": "g", "model": "m",
            "result": {"main_error_code": code, "justification_level": "J1", "response_complete": True,
                       "lines": lines}}


def _write(path, recs):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    return str(path)


def test_regression_set_is_20_dev_answers_covering_every_problem():
    import regression as R
    assert len(R.REGRESSION_IDS) == 20 and len(set(R.REGRESSION_IDS)) == 20
    assert all(i.startswith("A-") for i in R.REGRESSION_IDS)          # holdout(H-) 금지
    assert {i[:4] for i in R.REGRESSION_IDS} == {f"A-{n:02d}" for n in range(1, 11)}
    assert [r["answer_id"] for r in R.load_rows()] == R.REGRESSION_IDS


def test_regression_compare_flags_only_changed_fields(tmp_path):
    import regression as R
    ids = ["A-01-A", "A-01-C", "A-02-A"]
    base = _write(tmp_path / "b.jsonl", [_rec("A-01-A"), _rec("A-01-C", None, None), _rec("A-02-A", "R6", 3)])
    run = _write(tmp_path / "r.jsonl", [_rec("A-01-A"), _rec("A-01-C", None, None), _rec("A-02-A", "R6", 4)])
    cmp = R.compare(run, base, ids)
    assert cmp["changed"] == ["A-02-A"] and cmp["missing"] == []
    assert "| A-02-A | first_error | 3 | 4 |" in cmp["table"]
    assert "바뀐 답안: 1/3" in R.report(cmp, run, len(ids))


def test_regression_main_returns_nonzero_on_change_and_saves_baseline(tmp_path, monkeypatch):
    import regression as R
    monkeypatch.setattr(R, "REG_DIR", str(tmp_path))
    run = _write(tmp_path / "r.jsonl", [_rec(i) for i in R.REGRESSION_IDS])
    base = str(tmp_path / "baseline.jsonl")
    assert R.main(["--compare", run, "--save-baseline", "--baseline", base]) == 0 and os.path.exists(base)
    assert R.main(["--compare", run, "--baseline", base]) == 0
    changed = _write(tmp_path / "r2.jsonl", [_rec(i, outcome="review") for i in R.REGRESSION_IDS])
    assert R.main(["--compare", changed, "--baseline", base]) == 1


# ── API 사용량·비용 (Day 6) ───────────────────────────────


def test_call_cost_uses_model_price_and_date_suffix():
    import usage_cost as U
    u = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 1_000_000}
    assert U.call_cost("claude-opus-5-5", u) == pytest.approx(4 + 2 + 0.2)
    assert U.call_cost("claude-haiku-4-5-20251001", {"input_tokens": 2000, "output_tokens": 1000}) == pytest.approx(0.007)
    assert U.call_cost("unknown-model", u) is None and U.call_cost("claude-opus-5-5", None) is None


def test_record_cost_sums_calls_by_responding_model():
    import usage_cost as U
    rec = {"model": "claude-opus-5-5", "n_calls": 2, "raw_responses": [
        {"model": "claude-opus-5-5", "usage": {"input_tokens": 5000, "output_tokens": 1000}},
        {"model": "claude-opus-4-6", "usage": {"input_tokens": 5000, "output_tokens": 1000}}]}
    assert U.record_cost(rec) == pytest.approx(0.02 + 0.02 + 0.025 + 0.025)
    assert U.record_cost({"n_calls": 0, "raw_responses": []}) == 0.0
    rec["raw_responses"][1]["model"] = "mystery"
    assert U.record_cost(rec) is None          # 가격 모르는 호출이 있으면 추정하지 않음


def test_diagnose_records_cost():
    from tests.test_rv2_sanitize import FakeClient, _tool_block
    rec = g.diagnose(FakeClient([[_tool_block(good())]]), PID, ANSWER, model="claude-opus-5-5")
    assert rec["cost_usd"] == pytest.approx((10 * 4 + 5 * 20) / 1_000_000)   # 가짜 응답: 입력 10, 출력 5 토큰


def test_usage_summary_and_report(tmp_path):
    import usage_cost as U
    import usage_report
    recs = [{"n_calls": 1, "cost_usd": 0.05, "usage": {"input_tokens": 100, "output_tokens": 10}},
            {"n_calls": 2, "model": "claude-opus-5-5", "raw_responses": [
                {"usage": {"input_tokens": 1000, "output_tokens": 100}},
                {"usage": {"input_tokens": 1000, "output_tokens": 100}}],
             "usage": {"input_tokens": 2000, "output_tokens": 200}}]
    s = U.summarize(recs)
    assert s["records"] == 2 and s["calls"] == 3 and s["input_tokens"] == 2100
    assert s["cost_usd"] == pytest.approx(0.05 + 0.012, abs=1e-4)
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    assert "| r.jsonl | 2 | 3 | 2,100 | 210 | $0.06 |" in usage_report.main([str(p)])


def test_validate_rejects_r3_on_incomplete_answer():
    """중간에 멈춘(미완) 답안에는 R3를 붙이지 않는다 (grader 1.4)."""
    p = good()
    p.update(response_complete=False, main_error_code="R3")
    issues = g.validate(p, ANSWER, ["R3", "R8"], True)
    assert any("미완 답안" in i and "R3" in i for i in issues)
    p["response_complete"] = True
    assert not any("미완 답안" in i for i in g.validate(p, ANSWER, ["R3", "R8"], True))


def test_sanitize_error_criterion_to_int():
    """1.5: error_criterion도 matched_criterion처럼 정수로 정리하고, 없으면 칸을 만들지 않는다."""
    import rv2_grader
    res, _ = rv2_grader.sanitize({"lines": [
        {"no": 1, "student_text": "a", "judgment": "wrong", "matched_criterion": "4", "error_criterion": "①"},
        {"no": 2, "student_text": "b", "judgment": "ok", "matched_criterion": 2}],
        "math_result_correct": False, "input_status": "confirmed"}, mode="grade_only")
    assert res["lines"][0]["matched_criterion"] == 4 and res["lines"][0]["error_criterion"] == 1
    assert "error_criterion" not in res["lines"][1]
    res2, _ = rv2_grader.sanitize({"lines": [
        {"no": 1, "student_text": "a", "judgment": "wrong", "matched_criterion": 4, "violated_criterion": "1"}],
        "math_result_correct": False, "input_status": "confirmed"}, mode="grade_only")
    assert res2["lines"][0]["violated_criterion"] == 1   # 1.7: 코드북 v3.1 이름


def test_dataset_v31_prompt(tmp_path, monkeypatch):
    """R-v3.1 사본: 규칙 12 = 주 코드 선정, 미완 처리는 '답안 상태', R4 보류, 문항 2·7 기준표, 주의 항목."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import build_dataset_v31
    import rv2_codebook
    from prompts import rv2
    dst = tmp_path / "v31.xlsx"
    log = build_dataset_v31.build(dst=str(dst))
    assert log and all(row[5].startswith("코드북 R v3.1") for row in log)
    mats = rv2_codebook.load(str(dst))
    assert mats["codebook_version"] == "R-v3.1"
    rules = {str(r["번호"]): r for r in mats["prompt_rules"]}
    assert rules["12"]["규칙"].startswith("코드 후보가 둘 이상일 때") and rules["답안 상태"]["규칙"].startswith("중단된 답안")
    assert "R4" not in rv2.allowed_codes(mats, "INV-R-07")
    p7 = rv2.build_prompt(mats, "INV-R-07")["user_template"]
    assert "0이 아니기만 하면 된다고 보아 같은 부호 조건을 빠뜨리면 R2" in p7 and "- 주의: 포함 항목과 문구가 겹쳐도" in p7
    assert "(판정 규칙 '답안 상태')" in p7 and "(판정 규칙 12)" not in p7
    assert "c = 3/a로 읽으면 R6" in rv2.build_prompt(mats, "INV-R-02")["user_template"]
    base = rv2.build_prompt(rv2_codebook.load(), "INV-R-07")["user_template"]
    assert "(판정 규칙 12)" in base and "- 주의:" not in base   # 원래 자료(R-v3)로는 그대로
