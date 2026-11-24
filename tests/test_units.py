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
