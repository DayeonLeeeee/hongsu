# coding: utf-8
"""experiments/ 분석 스크립트 테스트 (LLM 호출 없음, 결과를 아는 가짜 실행 기록 사용)."""
import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import compare_runs  # noqa: E402
import ocr_compare  # noqa: E402
import score_experiments as se  # noqa: E402

ANS_COLS = ["answer_id", "problem_id", "유형", "학생_답안", "첫_독립_오류", "첫_독립_오류_기준", "error_code"]
ANSWERS = [
    # answer_id, 유형, 첫 줄(학생), 첫 줄(기준), 코드
    ("A1", "오답", "①", "②", "R8"),      # 학생 줄 번호와 기준표 번호가 다른 답안
    ("A2", "오답", "②", "②", "R6"),
    ("A3", "오탐검사", "없음", "없음", "없음"),
    ("A4", "오답", "없음", "없음", "없음"),  # 코드 없음 라벨
]


def rec(aid, rep, code, first_no=None, first_crit=None, *, review=False, complete=True, lines_code=None,
        error=None, extraction=("tool",), first_issues=(), n_calls=1):
    lines = []
    if first_no is not None:
        lines = [{"no": first_no, "judgment": "wrong", "matched_criterion": first_crit, "code": lines_code or code}]
    result = None if error else {
        "lines": lines, "main_error_code": code, "justification_level": "J1", "review_needed": review,
        "response_complete": complete, "missing_criteria": [] if complete else [4], "review_reason": "애매" if review else "",
    }
    r = {"answer_id": aid, "repeat": rep, "result": result, "error": error, "n_calls": n_calls,
         "extraction": list(extraction), "first_issues": list(first_issues), "validation_issues": [],
         "stop_reasons": ["end_turn"], "model": "m", "prompt_version": "p", "grader_version": "g", "codebook_version": "c"}
    import rv2_grader
    r["outcome"], r["review_sources"] = rv2_grader.classify_outcome(r)
    return r


@pytest.fixture()
def files(tmp_path):
    ans = tmp_path / "answers.csv"
    with open(ans, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(ANS_COLS)
        for aid, typ, fs, fc, code in ANSWERS:
            w.writerow([aid, "INV-R-01", typ, "x", fs, fc, code])
    results = [
        rec("A1", 1, "R8", first_no=1, first_crit=2),                     # 기준 번호로는 일치, 학생 번호로는 불일치
        rec("A2", 1, "R6", first_no=2, first_crit=2),
        rec("A3", 1, None),                                               # 오탐 없음
        rec("A4", 1, None, review=True),                                  # 코드 없음 라벨에 기권
        rec("A1", 2, None, review=True, extraction=("none", "text"), n_calls=2,
            first_issues=("결과 블록 없음 (stop_reason=end_turn)",)),        # 코드 있는 라벨에 기권 = 진짜 손실
        rec("A2", 2, None, first_no=3, first_crit=3, complete=False, lines_code="R3",
            first_issues=("line 3: quote가 답안 원문에 없음",), n_calls=2),
        rec("A3", 2, "R1", first_no=1, first_crit=1),                     # 오탐
        rec("A4", 2, None, error="boom"),
    ]
    res = tmp_path / "results.jsonl"
    res.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
    return str(ans), str(res), results


def test_exp_1_1_counts(files):
    _, _, results = files
    text = "\n".join(se.exp_1_1(results))
    assert "| 결과 출처: 도구(tool) / 텍스트(text) / 없음(none) | 7 / 1 / 1 |" in text
    assert "| 첫 호출에 형식 문제가 있던 행 | 1/8 (12.5%) |" in text
    assert "| 재시도한 행 (호출 2회) | 2/8 (25.0%) |" in text
    assert "| 형식 | 결과 블록 없음 | 1 |" in text
    assert "| 검증 | quote가 답안 원문에 없음 | 1 |" in text


def test_exp_1_2_basis_matters(files):
    ans, _, results = files
    answers = se.load_answers(ans)
    crit = "\n".join(se.exp_1_2(results, answers, "criterion", "첫_독립_오류_기준", []))
    stud = "\n".join(se.exp_1_2(results, answers, "student", "첫_독립_오류", []))
    # 라벨 첫 줄이 있는 행: A1×2, A2×2 → 4행 (error 행 없음). 기준: A1-1 일치, A2-1 일치, A2-2(3≠2) 불일치, A1-2 없음
    assert "| 첫 오류 줄 일치 (행 기준) | 2/4 (50.0%) |" in crit
    # 학생 줄 번호: A1-1 (1=1) 일치, A2-1 일치 → 같은 2/4지만 라벨 열이 달라 의미가 다름
    assert "| 첫 오류 줄 일치 (행 기준) | 2/4 (50.0%) |" in stud
    assert "| 오탐검사 답안에 코드가 붙은 행 | 1/2 (50.0%) |" in crit
    # 코드 일치: 7행(error 제외) 중 A1-1, A2-1, A3-1, A4-1(None=None) → 4/7
    assert "| 주 코드 일치 (행 기준, '없음'끼리도 일치) | 4/7 (57.1%) |" in crit


def test_exp_1_3_focus_rows(files):
    ans, _, results = files
    text = "\n".join(se.exp_1_2(results, se.load_answers(ans), "criterion", "첫_독립_오류_기준", ["A1", "ZZ"]))
    assert "## 1-3" in text and "| A1 (2회) | R8 | 2 | 없음 | 없음 | 예 | model |" in text
    assert "| ZZ | (결과 없음)" in text


def test_exp_1_5_scenarios(files):
    ans, _, results = files
    text = "\n".join(se.exp_1_5(results, se.load_answers(ans)))
    assert "| 기권 행 | 2 |" in text
    assert "(시나리오 1이면 일치) | 1 |" in text and "(진짜 손실) | 1 |" in text
    # 8행: 일치 = A1-1, A2-1, A3-1 (+ A4-1 기권은 시나리오 1에서만) / A2-2(None≠R6), A3-2, A4-2(error), A1-2(손실) 불일치
    assert "| 시나리오 1: '코드 없음' 라벨 기권 = 일치 | 4/8 (50.0%) |" in text
    assert "| 시나리오 2: '코드 없음' 라벨 기권 = 불일치 | 3/8 (37.5%) |" in text
    assert "| A4 | 1 | 없음 | 코드 없음 라벨 |" in text


def test_exp_1_6_incomplete_with_r3(files):
    ans, _, results = files
    text = "\n".join(se.exp_1_6(results, se.load_answers(ans)))
    assert "| 모델이 미완(response_complete=false)으로 본 행 | 1 |" in text
    assert "| └ 그중 R3가 붙은 행 (미완 처리 원칙상 붙으면 안 됨) | 1 |" in text


def test_exp_1_7_crosstab(files):
    ans, _, results = files
    text = "\n".join(se.exp_1_7(results, se.load_answers(ans), "R6"))
    assert "라벨이 R6인 행 2개 중 모델이 코드를 붙이지 않은 행(없음·기권): 1/2 (50.0%)" in text
    assert "| R8 | " in text and "기권" in text and "ERR" in text


def test_main_writes_report(files, tmp_path):
    ans, res, _ = files
    out = tmp_path / "r.md"
    se.main(["--results", res, "--answers", ans, "--first-col", "첫_독립_오류_기준",
             "--first-basis", "criterion", "--out", str(out)])
    t = out.read_text(encoding="utf-8")
    for h in ("## 1-1", "## 1-2", "## 1-3", "## 1-5", "## 1-6", "## 1-7"):
        assert h in t


def test_compare_runs(files, tmp_path):
    ans, res, _ = files
    after = tmp_path / "after.jsonl"
    after.write_text(json.dumps(rec("A1", 1, "R8", first_no=2, first_crit=2), ensure_ascii=False), encoding="utf-8")
    out = compare_runs.main(["--before", res, "--after", str(after), "--before-answers", ans,
                             "--after-answers", ans, "--ids", "A1"])
    assert "| A1 | 변환 후 | R8 | 2 | R8 | 2 |" in out
    assert "판정(코드·첫 줄)이 달라진 답안: 1/1" in out


def test_run_batch_skips_blank_answers(tmp_path):
    import run_batch
    p = tmp_path / "a.csv"
    p.write_text("answer_id,problem_id,학생_답안\nX1,INV-R-01,\nX2,INV-R-01,① f(1)=3\n", encoding="utf-8")
    assert [r["answer_id"] for r in run_batch.load_answers(str(p), False)] == ["X2"]


# ── OCR 비교 ──

def test_ocr_metrics():
    assert ocr_compare.cer("f(1) = 3", "f(1)=3") == 0
    assert ocr_compare.cer("abcd", "abxd") == 0.25
    assert ocr_compare.line_class("따라서 없다") == "한글"
    assert ocr_compare.line_class("f(-2)=5") == "수식"
    assert ocr_compare.line_class("① f(-2)=5이므로") == "혼합"
    ev = ocr_compare.evaluate(["① f(-2)=5", "따라서 없다"], ["① f(-2)=-5"], "f(-2)=5")
    assert ev["corrected"] is True and ev["n_pred_lines"] == 1
    assert ev["per_class"]["한글"] == [1.0]   # 빠진 줄은 오류율 1


def test_ocr_compare_end_to_end(tmp_path, monkeypatch):
    img = tmp_path / "s1.png"
    img.write_bytes(b"\x89PNG fake")
    samples = tmp_path / "samples.csv"
    samples.write_text('sample_id,image,problem_id,정답_텍스트,교정검사\n'
                       'S1,s1.png,INV-R-05,"① f(-2)=5\n② 따라서 없다",f(-2)=5\n', encoding="utf-8")
    monkeypatch.setitem(ocr_compare.ENGINES, "good", lambda b, m: ["① f(-2)=5", "② 따라서 없다"])
    monkeypatch.setitem(ocr_compare.ENGINES, "fixer", lambda b, m: ["① f(-2)=-5", "② 따라서 없다"])
    monkeypatch.setitem(ocr_compare.ENGINES, "nokey", lambda b, m: None)
    monkeypatch.setattr(ocr_compare, "grade_signature",
                        lambda pid, text: {"code": "R6" if "=5" in text and "=-5" not in text else None,
                                           "first_criterion": 1, "outcome": "graded", "error": None})
    out = tmp_path / "o.md"
    ocr_compare.main(["--samples", str(samples), "--engines", "good,fixer,nokey", "--grade", "--out", str(out)])
    t = out.read_text(encoding="utf-8")
    assert "| good | 1/1 | 0.000 |" in t and t.split("| good |")[1].split("\n")[0].rstrip().endswith("0/1 |")
    assert "| fixer | 1/1 |" in t and "| 1/1 | " in t.split("| fixer |")[1].split("\n")[0]
    assert "| nokey | 키 없음 |" in t
