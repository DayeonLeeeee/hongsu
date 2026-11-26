# coding: utf-8
"""experiments/ 분석 스크립트 테스트 (LLM 호출 없음, 결과를 아는 가짜 실행 기록 사용).

판정 정의는 키트 채점기(experiments/score_rv2.py)와 같아야 한다.
"""
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
import score_rv2 as K  # noqa: E402

EXPERT_COLS = ["answer_id", "problem_id", "유형", "학생_답안", "전문가_줄별판정", "전문가_첫오류", "전문가_코드", "전문가_J", "비고"]
HOLDOUT_COLS = ["answer_id", "블록번호", "problem_id", "상태", "학생_답안", "확정_주", "확정_보조", "확정_J", "그럴듯함", "조정·비고"]


def res(code=None, first_crit=None, *, review=False, complete=True, wrong_no_code=False, r3=False):
    lines = []
    if first_crit is not None or wrong_no_code:
        lines = [{"no": 1, "judgment": "wrong", "matched_criterion": first_crit,
                  "code": None if wrong_no_code else ("R3" if r3 else code)}]
    return {"lines": lines, "main_error_code": code, "justification_level": "J1", "review_needed": review,
            "response_complete": complete, "missing_criteria": [], "math_result_correct": False,
            "review_reason": "애매함" if review else ""}


def row(aid, rep, result, **kw):
    r = {"answer_id": aid, "repeat": rep, "result": result, "error": None if result else "boom",
         "n_calls": 1, "validation_issues": [], "stop_reasons": ["end_turn"],
         "model": "m", "prompt_version": "p", "codebook_version": "c"}
    r.update(kw)
    return r


def write_csv(path, cols, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture()
def expert(tmp_path):
    ans = tmp_path / "expert.csv"
    write_csv(ans, EXPERT_COLS + ["전문가_첫오류_통일"], [
        {"answer_id": "A1", "problem_id": "P", "유형": "오답", "전문가_첫오류": "①", "전문가_첫오류_통일": "②", "전문가_코드": "R8", "전문가_J": "J1"},
        {"answer_id": "A2", "problem_id": "P", "유형": "오답", "전문가_첫오류": "②", "전문가_첫오류_통일": "②", "전문가_코드": "R6", "전문가_J": "J1"},
        {"answer_id": "A3", "problem_id": "P", "유형": "오탐검사", "전문가_첫오류": "없음", "전문가_첫오류_통일": "없음", "전문가_코드": "없음", "전문가_J": "J2"},
    ])
    rows = [
        row("A1", 1, res("R8", 2)), row("A1", 2, res("R8", 2)), row("A1", 3, res(None, review=True)),
        row("A2", 1, res("R6", 2)), row("A2", 2, res("R6", 3)), row("A2", 3, res("R7", 2)),
        row("A3", 1, res()), row("A3", 2, res()), row("A3", 3, None),
    ]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(ans), str(p), rows


@pytest.fixture()
def holdout(tmp_path):
    ans = tmp_path / "holdout.csv"
    write_csv(ans, HOLDOUT_COLS, [
        {"answer_id": "H-01", "problem_id": "P", "상태": "유효", "확정_주": "R6", "확정_J": "J1"},
        {"answer_id": "H-02", "problem_id": "P", "상태": "유효", "확정_주": "코드없음", "확정_J": "J1"},
        {"answer_id": "H-03", "problem_id": "P", "상태": "유효", "확정_주": "코드없음", "확정_J": "J1"},
        {"answer_id": "H-04", "problem_id": "P", "상태": "유효", "확정_주": "정답", "확정_J": "J2"},
        {"answer_id": "H-05", "problem_id": "P", "상태": "유효", "확정_주": "코드없음", "확정_J": "J0",
         "조정·비고": "response_complete=false"},
        {"answer_id": "H-09", "problem_id": "P", "상태": "유효", "확정_주": "R4", "확정_J": "J1"},  # R-v3로 R2
        {"answer_id": "H-99", "problem_id": "P", "상태": "1차심사 미달", "확정_주": "R1", "확정_J": "J1"},
    ])
    rows = []
    for rep in (1, 2, 3):
        rows += [row("H-01", rep, res(None, review=True)),                  # R6 라벨에 기권 → 진짜 손실
                 row("H-02", rep, res(None, review=True)),                  # 코드 없음 라벨에 기권
                 row("H-03", rep, res(None, wrong_no_code=True)),           # 코드 없음 라벨에 코드누락
                 row("H-04", rep, res()),
                 row("H-05", rep, res(None, complete=False)),
                 row("H-09", rep, res("R2", 1)),
                 row("H-99", rep, res("R1", 1))]
    rows[0]["raw_responses"] = [{"content": [{"type": "thinking"}, {"type": "tool_use"}]}]
    rows[1]["raw_responses"] = [{"content": [{"type": "text", "text": "{}"}]}]
    p = tmp_path / "h.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(ans), str(p), rows


def run_report(argv):
    return se.main(argv)


def test_expert_first_line_matches_kit_definition(expert):
    ans, res_path, _ = expert
    t = run_report(["--results", res_path, "--answers", ans, "--focus-ids", "A1"])
    # 키트: 1회차 기준. A1 라벨 ①(=1) vs AI 2 → 불일치, A2 ②(=2) vs 2 → 일치
    assert "**첫 오류 줄 일치: 1/2 (50.0%)**" in t
    # 라벨 통일본 열을 쓰면 A1도 일치
    t2 = run_report(["--results", res_path, "--answers", ans, "--first-col", "전문가_첫오류_통일", "--only", "1-2"])
    assert "**첫 오류 줄 일치: 2/2 (100.0%)**" in t2
    assert "| A1 | R8 | 2 | 2 / 2 / 없음 | - / - / 없음 | R8 / R8 / 없음(기권) | 2/3 |" in t2


def test_expert_focus_and_vote(expert):
    ans, res_path, _ = expert
    t = run_report(["--results", res_path, "--answers", ans, "--focus-ids", "A1,ZZ", "--only", "1-3"])
    # A1: 기권은 표에서 빠지고 R8 다수결 → 일치. 1회차 오류 줄은 학생 1줄 → 기준 2
    assert "| A1 | R8 | 1 | R8 / R8 / 없음(기권) | R8 | 일치 | 1→기준2:R8 |" in t
    assert "| ZZ | (라벨 또는 결과 없음)" in t


def test_holdout_focus_default_and_latest(holdout, tmp_path):
    ans, res_path, rows = holdout
    old = tmp_path / "stage3_holdout45_m_c_20261101_0900.jsonl"
    new = tmp_path / "stage3_holdout45_m_c_20261124_1500.jsonl"
    old.write_text("", encoding="utf-8")
    new.write_text(open(res_path, encoding="utf-8").read(), encoding="utf-8")
    t = run_report(["--results", str(tmp_path / "stage3_holdout45_*.jsonl"), "--latest", "--answers", ans,
                    "--focus-ids", "H-01", "--only", "focus"])
    assert "stage3_holdout45_m_c_20261124_1500.jsonl" in t and "20261101" not in t
    assert "| H-01 | R6 | - | 없음(기권) / 없음(기권) / 없음(기권) | 없음(기권) | 불일치 | 없음 | 애매함 |" in t
    import score_experiments
    assert score_experiments.DEFAULT_FOCUS["holdout"] == "H-03,H-33,H-10,H-15"


def test_holdout_scenarios_and_relabel(holdout):
    ans, res_path, _ = holdout
    t = run_report(["--results", res_path, "--answers", ans, "--only", "1-5"])
    # 유효 6건 (H-99 제외). 키트 일치: H-04(정답, 오류없음), H-05(미완), H-09(R4→R2 재라벨, AI R2) = 3
    assert "| 키트 기준 (기권·코드누락 = 불일치) | 3/6 (50.0%) |" in t
    assert "| '코드 없음' 라벨에 대한 기권을 일치로 인정 | 4/6 (66.7%) |" in t          # + H-02
    assert "| '코드 없음' 라벨에 대한 기권 + 코드누락을 일치로 인정 | 5/6 (83.3%) |" in t  # + H-03
    assert "(진짜 손실): **1건**" in t and "| H-01 | R6 |" in t
    t2 = run_report(["--results", res_path, "--answers", ans, "--only", "1-5", "--no-v3-relabel"])
    assert "| 키트 기준 (기권·코드누락 = 불일치) | 2/5 (40.0%) |" in t2


def test_scenario_baseline_equals_kit_all_match(holdout):
    """시나리오 기준선이 키트 '전체 코드 일치'와 같은 계산인지 직접 대조."""
    ans, res_path, rows = holdout
    _, labels = K.load_labels(ans, True)
    runs = se.group_runs(rows, labels)
    kit_all = sum(int(K.is_match(K.vote([K.code_label(r["result"]) for _, r in runs[a]]), lab["code"]))
                  for a, lab in labels.items())
    assert f"| 키트 기준 (기권·코드누락 = 불일치) | {kit_all}/{len(labels)}" in "\n".join(se.exp_1_5(runs, labels))


def test_holdout_1_1_from_raw_responses(holdout):
    ans, res_path, _ = holdout
    t = run_report(["--results", res_path, "--answers", ans, "--only", "1-1"])
    assert "| 결과 출처: 도구 / 텍스트 / 없음 (호출 기준) | 1 / 1 / 0 |" in t
    assert "grader 1.3 이전 결과" in t


def test_1_1_with_first_issues(expert):
    rows = [row("A1", 1, res("R8", 2), extraction=["none", "tool"], n_calls=2,
                first_issues=["결과 블록 없음 (stop_reason=end_turn)", "line 3: 허용 밖 코드 R8"])]
    t = "\n".join(se.exp_1_1(rows))
    assert "| 결과 출처: 도구 / 텍스트 / 없음 (호출 기준) | 1 / 0 / 1 |" in t
    assert "| 첫 호출에 형식 문제가 있던 행 | 1/1 (100.0%) |" in t
    assert "| 검증 | 허용 밖 코드 R코드 | 1 |" in t


def test_holdout_incomplete_and_focus_code(holdout):
    ans, res_path, _ = holdout
    t = run_report(["--results", res_path, "--answers", ans, "--only", "1-6,1-7"])
    assert "| 라벨이 미완인 답안 | 1 |" in t and "- H-05: [False, False, False]" in t
    assert "라벨이 R6인 답안 1건 중 AI 다수결이 코드 없음: **1/1 (100.0%)**" in t


def test_compare_runs(expert, tmp_path):
    ans, res_path, _ = expert
    after = tmp_path / "after.jsonl"
    after.write_text("\n".join(json.dumps(row("A1", rep, res("R8", 2)), ensure_ascii=False) for rep in (1, 2, 3)),
                     encoding="utf-8")
    out = compare_runs.main(["--before", res_path, "--after", str(after), "--before-answers", ans,
                             "--after-answers", ans, "--ids", "A1,A2"])
    assert "| A1 | 변환 후 | R8 | 1 | R8 / R8 / R8 | R8 |" in out
    assert "| A2 | 변환 후 | (라벨 또는 결과 없음)" in out


def test_prepare_answers_from_xlsx_matches_kit_columns(tmp_path):
    import prepare_answers
    prepare_answers.main(["--out-dir", str(tmp_path)])
    with open(tmp_path / "expert_33.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 33 and list(rows[0].keys()) == (prepare_answers.KIT_COLS
                                                        + [prepare_answers.UNIFIED_COL, prepare_answers.UNIFIED_CODE_COL])
    assert all(r[prepare_answers.UNIFIED_COL] == r["전문가_첫오류"] for r in rows)
    assert all(r[prepare_answers.UNIFIED_CODE_COL] == r["전문가_코드"] for r in rows)
    kind, labels = K.load_labels(str(tmp_path / "expert_33.csv"), True)
    assert kind == "expert" and labels["A-01-A"]["code"] == "R8" and labels["A-01-A"]["first"] == 1
    with open(tmp_path / "converted_5.csv", encoding="utf-8-sig") as f:
        conv = list(csv.DictReader(f))
    assert [r["answer_id"] for r in conv] == prepare_answers.CONVERTED_IDS.split(",")
    assert all(r["학생_답안"] == "" and r["원래_답안"] for r in conv)
    assert all(r[prepare_answers.UNIFIED_CODE_COL] == r["전문가_코드"] for r in conv)


def test_code_col_marks_incomplete_and_changes_match(expert, tmp_path):
    """전문가_코드_통일에서 '미완'은 코드 없음 + 미완. 원래 라벨 열은 그대로 두고 통일 열만 읽는다."""
    import score_experiments as S
    src = expert[0]
    with open(src, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["전문가_코드_통일"] = "미완" if r["answer_id"] == rows[0]["answer_id"] else r.get("전문가_코드", "")
    p = tmp_path / "e.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _, orig = S.load_labels(str(p))
    _, uni = S.load_labels(str(p), code_col="전문가_코드_통일")
    aid = rows[0]["answer_id"]
    assert uni[aid]["code"] is None and uni[aid]["incomplete"] is True
    assert all(uni[a]["code"] == orig[a]["code"] for a in orig if a != aid)
    with pytest.raises(SystemExit):
        S.load_labels(str(src), code_col="없는열")
    assert S.load_labels(str(src), code_col="없는열", code_col_optional=True)[1] == S.load_labels(str(src))[1]


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
    assert ev["per_class"]["한글"] == [1.0]


def test_ocr_compare_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "s1.png").write_bytes(b"\x89PNG fake")
    samples = tmp_path / "samples.csv"
    samples.write_text('sample_id,image,problem_id,정답_텍스트,교정검사\n'
                       'S1,s1.png,INV-R-05,"① f(-2)=5\n② 따라서 없다",f(-2)=5\n', encoding="utf-8")
    monkeypatch.setitem(ocr_compare.ENGINES, "good", lambda b, m: ["① f(-2)=5", "② 따라서 없다"])
    monkeypatch.setitem(ocr_compare.ENGINES, "fixer", lambda b, m: ["① f(-2)=-5", "② 따라서 없다"])
    monkeypatch.setitem(ocr_compare.ENGINES, "nokey", lambda b, m: None)
    monkeypatch.setattr(ocr_compare, "grade_signature",
                        lambda pid, text: {"code": "R6" if "=-5" not in text else None,
                                           "first_criterion": 1, "outcome": "graded", "error": None})
    out = tmp_path / "o.md"
    ocr_compare.main(["--samples", str(samples), "--engines", "good,fixer,nokey", "--grade", "--out", str(out)])
    t = out.read_text(encoding="utf-8")
    good = t.split("| good |")[1].split("\n")[0]
    fixer = t.split("| fixer |")[1].split("\n")[0]
    assert good.startswith(" 1/1 | 0.000 |") and good.rstrip().endswith("0/1 |")
    assert "| 1/1 | " in fixer and fixer.rstrip().endswith("1/1 |")
    assert "| nokey | 키 없음 |" in t


def test_compare_runs_all_ids(expert, tmp_path):
    ans, res_path, _ = expert
    out = compare_runs.main(["--before", res_path, "--after", res_path, "--before-answers", ans,
                             "--after-answers", ans, "--ids", "all", "--before-tag", "1.2", "--after-tag", "1.3",
                             "--title", "1-1b 재채점 비교"])
    assert out.startswith("# 1-1b 재채점 비교") and "다수결 판정이 달라진 답안: 0/3" in out
    assert "| A1 | 1.3 |" in out and "오세준" not in out


def test_compare_runs_ignores_one_sided_answers(expert, tmp_path):
    ans, res_path, _ = expert
    after = tmp_path / "after.jsonl"
    after.write_text(json.dumps(row("A1", 1, res("R6", 2)), ensure_ascii=False), encoding="utf-8")
    out = compare_runs.main(["--before", res_path, "--after", str(after), "--before-answers", ans,
                             "--after-answers", ans, "--ids", "A1,A2,A3"])
    assert "다수결 판정이 달라진 답안: 1/1 (한쪽 결과가 없는 답안 2건 제외)" in out


def test_run_py_flow(tmp_path, monkeypatch, expert, holdout):
    """setup 없이 실행하면 안내 후 중단, 키트 위치를 기억하면 baseline이 돈다."""
    import run as R
    monkeypatch.setattr(R, "ANSWERS", str(tmp_path / "answers"))
    monkeypatch.setattr(R, "KIT_PATH_FILE", str(tmp_path / "answers" / ".kit_path"))
    monkeypatch.setattr(R, "OUT", str(tmp_path / "out"))
    monkeypatch.delenv("RV2_KIT", raising=False)
    with pytest.raises(SystemExit):
        R.main(["status"])
    kit = tmp_path / "kit"
    (kit / "results").mkdir(parents=True)
    (kit / "answers").mkdir()
    import shutil
    shutil.copy(expert[0], kit / "answers" / "expert_33.csv")
    shutil.copy(holdout[0], kit / "answers" / "holdout45.csv")
    shutil.copy(expert[1], kit / "results" / R.KIT_EXPERT_1111)
    shutil.copy(holdout[1], kit / "results" / R.KIT_HOLDOUT_1111)
    (kit / "score_rv2.py").write_text("", encoding="utf-8")   # 키트 표식
    R.main(["setup", "--kit", str(kit)])
    assert (tmp_path / "answers" / ".kit_path").read_text(encoding="utf-8").strip() == str(kit)
    R.main(["baseline"])
    outs = sorted(os.listdir(tmp_path / "out"))
    assert len(outs) == 2 and outs[0].startswith("baseline_expert33") and outs[1].startswith("baseline_holdout45")


def test_run_py_finds_nested_kit_and_rejects_repo(tmp_path, monkeypatch):
    """zip을 '압축 풀기'하면 생기는 rv2_kit/rv2_kit는 안쪽을 찾고, 키트가 아닌 폴더는 멈춘다."""
    import run as R
    inner = tmp_path / "rv2_kit" / "rv2_kit"
    (inner / "answers").mkdir(parents=True)
    (inner / "answers" / "expert_33.csv").write_text("answer_id\n", encoding="utf-8")
    (inner / "score_rv2.py").write_text("", encoding="utf-8")
    assert R.find_kit(str(tmp_path / "rv2_kit")) == str(inner)
    assert R.find_kit(str(inner)) == str(inner)
    repo_like = tmp_path / "repo"
    (repo_like / "answers").mkdir(parents=True)
    (repo_like / "answers" / "expert_33.csv").write_text("answer_id\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        R.find_kit(str(repo_like))


def test_run_py_fixes_excel_saved_csv(tmp_path, monkeypatch):
    """엑셀 cp949 저장: UTF-8로 바꾸고, 키트에서 온 칸('?'로 깨진 학생_답안, -1이 된 라벨)은 키트 원본으로 되돌리되
    사람이 고친 '전문가_첫오류_통일'은 둔다. 사람이 쓴 통일 칸이 -1이면 멈춘다."""
    import run as R
    kit = tmp_path / "kit"
    (kit / "answers").mkdir(parents=True)
    (kit / "answers" / "expert_33.csv").write_text(
        "answer_id,학생_답안,전문가_첫오류\nA-04-A,f⁻¹(5)=2,(1)\n", encoding="utf-8-sig")
    ans = tmp_path / "answers"
    ans.mkdir()
    monkeypatch.setattr(R, "ANSWERS", str(ans))
    monkeypatch.setattr(R, "ROOT", str(tmp_path))
    monkeypatch.setattr(R, "KIT_PATH_FILE", str(ans / ".kit_path"))
    monkeypatch.delenv("RV2_KIT", raising=False)
    (ans / ".kit_path").write_text(str(kit), encoding="utf-8")
    p = ans / "expert_33.csv"
    p.write_bytes("answer_id,학생_답안,전문가_첫오류,전문가_첫오류_통일\nA-04-A,f?¹(5)=2,-1,2\n".encode("cp949"))
    R.check_answer_csvs()
    assert (ans / "expert_33.csv.cp949.bak").exists()
    row = R.read_csv(str(p))[0]
    assert row["학생_답안"] == "f⁻¹(5)=2" and row["전문가_첫오류"] == "(1)" and row["전문가_첫오류_통일"] == "2"
    p.write_text("answer_id,학생_답안,전문가_첫오류,전문가_첫오류_통일\nA-04-A,f⁻¹(5)=2,(1),-2\n", encoding="utf-8-sig")
    with pytest.raises(SystemExit):
        R.check_answer_csvs()
