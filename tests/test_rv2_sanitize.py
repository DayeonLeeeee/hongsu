# coding: utf-8
"""rv2_grader 응답 정제·추출·재시도 테스트 (LLM 호출 없음, 가짜 클라이언트 사용).

실행: 저장소 루트에서  python -m pytest tests -q
"""
import copy
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rv2_grader as g  # noqa: E402

PID = "INV-R-01"  # materials/dataset_rv2.xlsx에 있는 문항, 허용 코드 R8
ANSWER = "1) f(1)=a\n2) f^(-1)(a)=2"

GOOD = {
    "lines": [
        {"no": 1, "student_text": "f(1)=a", "judgment": "ok", "code": None, "quote": None, "reason": None},
        {"no": 2, "student_text": "f^(-1)(a)=2", "judgment": "wrong", "code": "R8",
         "quote": "f^(-1)(a)=2", "reason": "좌표 교환 연결 실패"},
    ],
    "math_result_correct": False,
    "input_status": "confirmed",
    "main_error_code": "R8",
    "justification_level": "J1",
    "review_needed": False,
    "review_reason": "",
    "response_complete": True,
    "missing_criteria": [],
}


def good():
    return copy.deepcopy(GOOD)


# ── sanitize ──────────────────────────────────────────────

def test_clean_response_has_no_issues():
    out, issues = g.sanitize(good())
    assert issues == []
    assert out["main_error_code"] == "R8"
    assert len(out["lines"]) == 2


def test_line_as_json_string_is_recovered():
    p = good()
    p["lines"][1] = json.dumps(p["lines"][1], ensure_ascii=False)
    out, issues = g.sanitize(p)
    assert issues == []
    assert out["lines"][1]["code"] == "R8"


def test_plain_string_line_is_reported_not_silently_dropped():
    p = good()
    p["lines"].append("3번 줄: 결론 누락")
    out, issues = g.sanitize(p)
    assert len(out["lines"]) == 2
    assert any("줄이 객체가 아님" in s for s in issues)


def test_string_booleans_are_converted():
    p = good()
    p["math_result_correct"] = "false"
    p["review_needed"] = "true"
    p["response_complete"] = "True"
    out, issues = g.sanitize(p)
    assert issues == []
    assert out["math_result_correct"] is False
    assert out["review_needed"] is True
    assert out["response_complete"] is True


def test_unparseable_math_result_is_reported():
    p = good()
    p["math_result_correct"] = "모름"
    out, issues = g.sanitize(p)
    assert out["math_result_correct"] is None
    assert any("math_result_correct" in s for s in issues)


def test_null_words_become_none_and_codes_uppercase():
    p = good()
    p["main_error_code"] = "null"
    p["lines"][0]["code"] = ""
    p["lines"][1]["code"] = "r8"
    out, issues = g.sanitize(p)
    assert out["main_error_code"] is None
    assert out["lines"][0]["code"] is None
    assert out["lines"][1]["code"] == "R8"


def test_judgment_case_and_aliases():
    p = good()
    p["lines"][0]["judgment"] = "OK"
    p["lines"][1]["judgment"] = "Wrong"
    out, issues = g.sanitize(p)
    assert issues == []
    assert [l["judgment"] for l in out["lines"]] == ["ok", "wrong"]


def test_unknown_judgment_is_reported():
    p = good()
    p["lines"][0]["judgment"] = "maybe"
    _, issues = g.sanitize(p)
    assert any("judgment" in s for s in issues)


def test_circled_line_number():
    p = good()
    p["lines"][1]["no"] = "②"
    out, issues = g.sanitize(p)
    assert issues == []
    assert out["lines"][1]["no"] == 2


def test_lines_not_a_list():
    p = good()
    p["lines"] = "1번 줄 ok, 2번 줄 wrong"
    out, issues = g.sanitize(p)
    assert out["lines"] == []
    assert any("lines가 리스트가 아님" in s for s in issues)


def test_justification_level_variants():
    p = good()
    p["justification_level"] = "j2"
    out, _ = g.sanitize(p)
    assert out["justification_level"] == "J2"
    p["justification_level"] = "not applicable"
    out, _ = g.sanitize(p)
    assert out["justification_level"] == "not_applicable"
    p["justification_level"] = "J9"
    _, issues = g.sanitize(p)
    assert any("justification_level" in s for s in issues)


def test_whole_response_as_json_string():
    out, issues = g.sanitize(json.dumps(good(), ensure_ascii=False))
    assert issues == []
    assert out["main_error_code"] == "R8"


def test_non_dict_response():
    out, issues = g.sanitize(["not", "a", "dict"])
    assert out == {}
    assert issues


# ── extract_result ────────────────────────────────────────

def _tool_block(d):
    return SimpleNamespace(type="tool_use", input=d, name="x", id="t1")


def _text_block(t):
    return SimpleNamespace(type="text", text=t)


def test_extract_tool_block():
    d, src = g.extract_result([_tool_block(good())])
    assert src == "tool" and d["main_error_code"] == "R8"


def test_extract_fenced_json_text():
    t = "판정 결과입니다.\n```json\n" + json.dumps(good(), ensure_ascii=False) + "\n```"
    d, src = g.extract_result([_text_block(t)])
    assert src == "text" and d["main_error_code"] == "R8"


def test_extract_bare_json_in_prose():
    t = "아래와 같이 판정합니다: " + json.dumps(good(), ensure_ascii=False) + " 이상입니다."
    d, src = g.extract_result([_text_block(t)])
    assert src == "text" and len(d["lines"]) == 2


def test_extract_nothing():
    d, src = g.extract_result([_text_block("애매해서 판정을 보류합니다.")])
    assert d is None and src == "none"


# ── diagnose (재시도 흐름) ─────────────────────────────────

class FakeClient:
    """미리 정한 응답 블록 목록을 차례로 돌려준다."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        blocks = self.responses.pop(0)
        return SimpleNamespace(content=blocks, stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                                     cache_creation_input_tokens=0,
                                                     cache_read_input_tokens=0))


def test_no_block_then_valid_retries_once():
    c = FakeClient([[_text_block("판정 보류")], [_tool_block(good())]])
    rec = g.diagnose(c, PID, ANSWER, temperature=None)
    assert rec["error"] is None
    assert rec["n_calls"] == 2
    assert rec["extraction"] == ["none", "tool"]
    assert rec["first_issues"] and rec["validation_issues"] == []
    assert rec["result"]["main_error_code"] == "R8"


def test_string_line_triggers_retry():
    bad = good()
    bad["lines"].append("3번 줄 설명")
    c = FakeClient([[_tool_block(bad)], [_tool_block(good())]])
    rec = g.diagnose(c, PID, ANSWER, temperature=None)
    assert rec["n_calls"] == 2
    assert any("줄이 객체가 아님" in s for s in rec["first_issues"])
    assert rec["validation_issues"] == []
    assert "[이전 출력의 문제점" in c.calls[1]["messages"][0]["content"]


def test_no_block_twice_sets_error():
    c = FakeClient([[_text_block("보류")], [_text_block("보류")]])
    rec = g.diagnose(c, PID, ANSWER, temperature=None)
    assert rec["result"] is None
    assert rec["error"] and "결과 블록 없음" in rec["error"]


def test_issues_remaining_after_retry_are_kept():
    bad = good()
    bad["lines"][1]["quote"] = "답안에 없는 문장"
    c = FakeClient([[_tool_block(bad)], [_tool_block(copy.deepcopy(bad))]])
    rec = g.diagnose(c, PID, ANSWER, temperature=None)
    assert rec["error"] is None
    assert any("quote가 답안 원문에 없음" in s for s in rec["validation_issues"])


def test_tool_choice_auto_and_temperature_omitted():
    c = FakeClient([[_text_block("```json\n" + json.dumps(good(), ensure_ascii=False) + "\n```")]])
    rec = g.diagnose(c, PID, ANSWER, temperature=None, tool_choice="auto")
    assert c.calls[0]["tool_choice"] == {"type": "auto"}
    assert "temperature" not in c.calls[0]
    assert rec["extraction"] == ["text"]
    assert rec["error"] is None


def test_tool_choice_default_forces_tool():
    c = FakeClient([[_tool_block(good())]])
    g.diagnose(c, PID, ANSWER, temperature=None, tool_choice="tool")
    assert c.calls[0]["tool_choice"]["type"] == "tool"


def test_grader_version_recorded():
    c = FakeClient([[_tool_block(good())]])
    rec = g.diagnose(c, PID, ANSWER, temperature=None)
    assert rec["grader_version"] == g.GRADER_VERSION


# ── classify_outcome ──────────────────────────────────────

def _rec(result=None, **kw):
    base = {"result": result, "error": None, "validation_issues": [], "stop_reasons": ["end_turn"]}
    base.update(kw)
    return base


def test_outcome_graded():
    assert g.classify_outcome(_rec(good())) == ("graded", [])


def test_outcome_model_abstained():
    r = good()
    r["review_needed"] = True
    assert g.classify_outcome(_rec(r)) == ("review", ["model"])


def test_outcome_validation_and_truncation():
    out = g.classify_outcome(_rec(good(), validation_issues=["x"], stop_reasons=["max_tokens"]))
    assert out == ("review", ["validation", "max_tokens"])


def test_outcome_incomplete():
    r = good()
    r["response_complete"] = False
    assert g.classify_outcome(_rec(r)) == ("incomplete", [])


def test_outcome_review_wins_over_incomplete():
    r = good()
    r["response_complete"] = False
    r["review_needed"] = True
    assert g.classify_outcome(_rec(r))[0] == "review"


def test_outcome_blank_and_error():
    r = good()
    r["input_status"] = "not_attempted"
    assert g.classify_outcome(_rec(r)) == ("not_attempted", [])
    assert g.classify_outcome(_rec(None, error="boom")) == ("error", ["error"])
