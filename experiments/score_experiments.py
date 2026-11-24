# coding: utf-8
"""
미팅 전 필수 실험 분석 (1-1, 1-2, 1-3, 1-5, 1-6, 1-7).

run_batch.py가 만든 결과 JSONL과 라벨이 있는 답안 CSV를 받아 실험별 표를 만든다.
LLM을 부르지 않는다 (비용 0). 채점은 run_batch.py로 먼저 돌린다.

사용:
  python experiments/score_experiments.py \
      --results results/stage3_expert_33_....jsonl \
      --answers answers/expert_33.csv \
      --first-col 첫_독립_오류_기준 --first-basis criterion \
      --out experiments/out/expert33_report.md

  * --results는 여러 개 줄 수 있다 (같은 답안 파일을 나눠 돌린 경우).
  * 라벨 열 이름은 옵션으로 바꿀 수 있다 (holdout CSV 열 이름이 다르면 맞춘다).

용어:
  - 행(run) = 답안 1건 × 반복 1회. 일치율은 행 기준으로 세고, 반복별로도 나눠 보여 준다.
  - 첫 오류 줄(모델) = judgment가 wrong인 줄 중 번호가 가장 작은 줄.
      --first-basis student   → 그 줄의 학생 줄 번호(no)와 라벨 비교
      --first-basis criterion → 그 줄의 기준표 번호(matched_criterion)와 라벨 비교
  - 기권 = 모델이 검수 요청(review_needed)을 하고 주 코드를 비운 경우
           (outcome=review, review_sources에 model, main_error_code 없음).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import rv2_grader  # noqa: E402  (classify_outcome, _to_int 재사용)

NONE_LABELS = {"", "없음", "none", "null", "-", "해당없음", "해당 없음"}

# 정제(형식) 단계에서 나오는 사유. 이 밖의 사유는 검증(내용: 인용·허용 코드·기준 번호 등).
SCHEMA_ISSUE_KEYS = ["결과 블록 없음", "응답이 객체", "lines 없음", "lines가 리스트", "줄이 객체가 아님",
                     "줄 번호(no)", "judgment 값", "math_result_correct 해석", "input_status 값",
                     "justification_level"]


# ─────────────────────────────────────
# 입력
# ─────────────────────────────────────

def load_results(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_answers(path, include_all=False):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and "상태" in rows[0] and not include_all:
        rows = [r for r in rows if (r.get("상태") or "").strip() == "유효"]
    return {r["answer_id"]: r for r in rows}


def label_code(v):
    v = (v or "").strip()
    return None if v.lower() in NONE_LABELS else v.upper()


def label_first(v):
    v = (v or "").strip()
    if v.lower() in NONE_LABELS:
        return None
    return rv2_grader._to_int(v)


# ─────────────────────────────────────
# 결과 한 행 해석
# ─────────────────────────────────────

def model_view(rec, basis):
    """결과 기록 → 비교에 쓰는 값들."""
    res = rec.get("result") or {}
    outcome = rec.get("outcome")
    sources = rec.get("review_sources")
    if outcome is None:  # 1.3 이전 결과 파일
        outcome, sources = rv2_grader.classify_outcome(rec)
    sources = sources or []
    code = res.get("main_error_code") or None
    wrong = sorted((ln for ln in (res.get("lines") or []) if isinstance(ln, dict) and ln.get("judgment") == "wrong"),
                   key=lambda ln: (ln.get("no") is None, ln.get("no") or 0))
    first = None
    if wrong:
        first = wrong[0].get("matched_criterion") if basis == "criterion" else wrong[0].get("no")
        first = rv2_grader._to_int(first)
    any_code = bool(code) or any(isinstance(ln, dict) and ln.get("code") for ln in (res.get("lines") or []))
    return {
        "error": rec.get("error"),
        "outcome": outcome,
        "sources": sources,
        "code": code,
        "first": first,
        "has_wrong": bool(wrong),
        "any_code": any_code,
        "J": res.get("justification_level"),
        "complete": res.get("response_complete"),
        "missing": res.get("missing_criteria") or [],
        "abstain": outcome == "review" and "model" in sources and not code,
    }


def _issue_kind(msg):
    """사유 문장 → (형식|검증, 묶음 이름). 줄 번호와 뒤에 붙은 값은 떼서 같은 종류끼리 센다."""
    for k in SCHEMA_ISSUE_KEYS:
        if k in msg:
            return "형식", k
    m = re.sub(r"^(lines?\s*\[?#?\w*\]?)\s*:\s*", "", msg.strip())   # 앞의 "line 3: "
    m = m.split(":", 1)[0].strip()                                         # 뒤의 ": 값"
    m = re.sub(r"\bR\d+\b", "R코드", m)
    return "검증", m[:50]


def _pct(a, b):
    return f"{a}/{b} ({a / b * 100:.1f}%)" if b else f"{a}/0"


# ─────────────────────────────────────
# 실험별 분석
# ─────────────────────────────────────

def exp_1_1(results):
    """스키마 안정성: 결과를 어디서 읽었는지, 첫 호출에서 형식 문제가 얼마나 났는지."""
    n = len(results)
    calls = sum(r.get("n_calls") or 0 for r in results)
    extraction = Counter(x for r in results for x in (r.get("extraction") or []))
    retried = sum(1 for r in results if (r.get("n_calls") or 0) > 1)
    errors = sum(1 for r in results if r.get("error"))
    truncated = sum(1 for r in results if "max_tokens" in (r.get("stop_reasons") or []))
    kinds = Counter()
    schema_rows = 0
    for r in results:
        fi = r.get("first_issues") or []
        ks = [_issue_kind(m) for m in fi]
        kinds.update(ks)
        if any(k[0] == "형식" for k in ks):
            schema_rows += 1
    remaining = sum(1 for r in results if r.get("validation_issues"))
    has_fields = any("extraction" in r for r in results)

    out = ["## 1-1 스키마 안정성", ""]
    if not has_fields:
        out += ["> 이 결과 파일에는 extraction/first_issues 필드가 없다 (grader 1.3 이전 실행). "
                "1.3 코드로 다시 돌려야 측정할 수 있다.", ""]
    out += [
        "| 항목 | 값 |", "| --- | --- |",
        f"| 실행 행 수 (답안 × 반복) | {n} |",
        f"| 총 API 호출 | {calls} |",
        f"| 결과 출처: 도구(tool) / 텍스트(text) / 없음(none) | {extraction.get('tool', 0)} / {extraction.get('text', 0)} / {extraction.get('none', 0)} |",
        f"| 첫 호출에 형식 문제가 있던 행 | {_pct(schema_rows, n)} |",
        f"| 재시도한 행 (호출 2회) | {_pct(retried, n)} |",
        f"| 재시도 후에도 문제가 남은 행 | {_pct(remaining, n)} |",
        f"| 결과 없음(error) | {_pct(errors, n)} |",
        f"| 출력 잘림(max_tokens) | {_pct(truncated, n)} |",
        "",
    ]
    if kinds:
        out += ["첫 호출 사유 (형식 = 정제 단계, 검증 = 인용·코드·기준 번호 등 내용 검사)", "",
                "| 구분 | 사유 | 건수 |", "| --- | --- | --- |"]
        out += [f"| {k[0]} | {k[1]} | {c} |" for k, c in kinds.most_common()]
        out.append("")
    return out


def exp_1_2(results, answers, basis, first_col, focus_ids):
    """라벨 통일 후 첫 오류 줄 일치 + 코드 일치 + 오탐. 1-3은 focus_ids 행만 따로."""
    per_answer = defaultdict(list)
    first_match = first_total = code_match = code_total = 0
    fp = fp_total = 0
    by_rep = defaultdict(lambda: [0, 0])
    for r in results:
        a = answers.get(r.get("answer_id"))
        if not a:
            continue
        mv = model_view(r, basis)
        lc, lf = label_code(a.get("error_code")), label_first(a.get(first_col))
        per_answer[r["answer_id"]].append((r.get("repeat"), mv, lc, lf))
        if mv["error"]:
            continue
        code_total += 1
        code_match += int(mv["code"] == lc)
        if lf is not None:
            first_total += 1
            ok = mv["first"] == lf
            first_match += int(ok)
            by_rep[r.get("repeat")][0] += int(ok)
            by_rep[r.get("repeat")][1] += 1
        if (a.get("유형") or "").strip() == "오탐검사":
            fp_total += 1
            fp += int(mv["any_code"])

    basis_txt = "기준표 번호(matched_criterion)" if basis == "criterion" else "학생 줄 번호(no)"
    out = ["## 1-2 첫 오류 줄 일치", "",
           f"라벨 열 `{first_col}`을 모델의 {basis_txt}와 비교. 라벨이 '없음'인 답안은 제외.", "",
           "| 항목 | 값 |", "| --- | --- |",
           f"| 첫 오류 줄 일치 (행 기준) | {_pct(first_match, first_total)} |",
           f"| 주 코드 일치 (행 기준, '없음'끼리도 일치) | {_pct(code_match, code_total)} |",
           f"| 오탐검사 답안에 코드가 붙은 행 | {_pct(fp, fp_total)} |", ""]
    if by_rep:
        out += ["반복별 첫 오류 줄 일치: " + ", ".join(f"{k}회차 {v[0]}/{v[1]}" for k, v in sorted(by_rep.items(), key=lambda x: str(x[0]))), ""]

    out += ["답안별 (반복마다 모델 값을 순서대로 표시)", "",
            "| answer_id | 라벨 코드 | 라벨 첫 줄 | 모델 코드 | 모델 첫 줄 | 첫 줄 일치 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for aid in sorted(per_answer):
        runs = sorted(per_answer[aid], key=lambda x: str(x[0]))
        lc, lf = runs[0][2], runs[0][3]
        codes = " / ".join("ERR" if m["error"] else (m["code"] or "없음") for _, m, _, _ in runs)
        firsts = " / ".join("ERR" if m["error"] else str(m["first"] if m["first"] is not None else "없음") for _, m, _, _ in runs)
        hit = "-" if lf is None else f"{sum(1 for _, m, _, _ in runs if not m['error'] and m['first'] == lf)}/{len(runs)}"
        out.append(f"| {aid} | {lc or '없음'} | {lf if lf is not None else '없음'} | {codes} | {firsts} | {hit} |")
    out.append("")

    if focus_ids:
        out += ["## 1-3 지정 답안 재확인", "", "| answer_id | 라벨 코드 | 라벨 첫 줄 | 모델 코드 | 모델 첫 줄 | 기권 | 검수 사유 |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        for aid in focus_ids:
            runs = sorted(per_answer.get(aid, []), key=lambda x: str(x[0]))
            if not runs:
                out.append(f"| {aid} | (결과 없음) | | | | | |")
                continue
            for rep, m, lc, lf in runs:
                out.append(f"| {aid} ({rep}회) | {lc or '없음'} | {lf if lf is not None else '없음'} | "
                           f"{m['code'] or '없음'} | {m['first'] if m['first'] is not None else '없음'} | "
                           f"{'예' if m['abstain'] else ''} | {', '.join(m['sources'])} |")
        out.append("")
    return out


def exp_1_5(results, answers):
    """기권 손실 분류 + '코드 없음 라벨에 기권 = 일치'로 볼 때와 아닐 때의 코드 일치율."""
    rows = []
    for r in results:
        a = answers.get(r.get("answer_id"))
        if not a:
            continue
        mv = model_view(r, "student")
        rows.append((r, a, mv, label_code(a.get("error_code"))))

    def agree(scenario):
        hit = tot = 0
        by_rep = defaultdict(lambda: [0, 0])
        for r, a, mv, lc in rows:
            tot += 1
            if mv["error"]:
                ok = False
            elif mv["abstain"]:
                ok = scenario == 1 and lc is None
            else:
                ok = mv["code"] == lc
            hit += int(ok)
            by_rep[r.get("repeat")][0] += int(ok)
            by_rep[r.get("repeat")][1] += 1
        return hit, tot, by_rep

    abst = [(r, a, mv, lc) for r, a, mv, lc in rows if mv["abstain"]]
    on_none = [x for x in abst if x[3] is None]
    on_code = [x for x in abst if x[3] is not None]
    decided = [(r, a, mv, lc) for r, a, mv, lc in rows if not mv["abstain"] and not mv["error"]]
    dec_hit = sum(1 for _, _, mv, lc in decided if mv["code"] == lc)
    s1, t1, rep1 = agree(1)
    s2, t2, rep2 = agree(2)

    out = ["## 1-5 기권 손실 분류", "",
           "기권 = 모델이 검수를 요청하고 주 코드를 비운 경우.", "",
           "| 항목 | 값 |", "| --- | --- |",
           f"| 전체 행 | {len(rows)} |",
           f"| 기권 행 | {len(abst)} |",
           f"| └ 라벨이 '코드 없음'인 답안에 기권 (시나리오 1이면 일치) | {len(on_none)} |",
           f"| └ 라벨에 코드가 있는 답안에 기권 (진짜 손실) | {len(on_code)} |",
           f"| 판정을 내린 행만의 코드 일치 | {_pct(dec_hit, len(decided))} |",
           f"| 시나리오 1: '코드 없음' 라벨 기권 = 일치 | {_pct(s1, t1)} |",
           f"| 시나리오 2: '코드 없음' 라벨 기권 = 불일치 | {_pct(s2, t2)} |", ""]
    if rep1:
        out += ["반복별: " + ", ".join(f"{k}회차 시나리오1 {rep1[k][0]}/{rep1[k][1]} · 시나리오2 {rep2[k][0]}/{rep2[k][1]}"
                                     for k in sorted(rep1, key=str)), ""]
    if abst:
        out += ["기권 행 목록 (교수님 재문의 메시지에 넣을 답안 ID)", "",
                "| answer_id | 회차 | 라벨 코드 | 분류 | 모델 검수 사유 |", "| --- | --- | --- | --- | --- |"]
        for r, a, mv, lc in sorted(abst, key=lambda x: (x[0]["answer_id"], str(x[0].get("repeat")))):
            reason = ((r.get("result") or {}).get("review_reason") or "").replace("|", "/").replace("\n", " ")[:80]
            out.append(f"| {r['answer_id']} | {r.get('repeat')} | {lc or '없음'} | "
                       f"{'코드 없음 라벨' if lc is None else '진짜 손실'} | {reason} |")
        out.append("")
    return out


def exp_1_6(results, answers):
    """미완 답안: 모델이 미완으로 본 행, 그중 R3가 붙은 행, 라벨이 R3인 답안."""
    inc = []
    r3_any = []
    for r in results:
        a = answers.get(r.get("answer_id"))
        mv = model_view(r, "student")
        res = r.get("result") or {}
        has_r3 = mv["code"] == "R3" or any(isinstance(ln, dict) and (ln.get("code") or "").upper() == "R3"
                                          for ln in (res.get("lines") or []))
        if mv["complete"] is False:
            inc.append((r, a, mv, has_r3))
        if has_r3:
            r3_any.append(r)
    label_r3 = sorted(aid for aid, a in answers.items() if label_code(a.get("error_code")) == "R3")
    out = ["## 1-6 미완 답안", "",
           "| 항목 | 값 |", "| --- | --- |",
           f"| 모델이 미완(response_complete=false)으로 본 행 | {len(inc)} |",
           f"| └ 그중 R3가 붙은 행 (미완 처리 원칙상 붙으면 안 됨) | {sum(1 for x in inc if x[3])} |",
           f"| R3가 붙은 전체 행 | {len(r3_any)} |",
           f"| 라벨이 R3인 답안 | {', '.join(label_r3) or '없음'} |", ""]
    if inc:
        out += ["| answer_id | 회차 | 결과 | 빠진 기준 | R3 | 라벨 코드 |", "| --- | --- | --- | --- | --- | --- |"]
        for r, a, mv, has_r3 in sorted(inc, key=lambda x: (x[0]["answer_id"], str(x[0].get("repeat")))):
            out.append(f"| {r['answer_id']} | {r.get('repeat')} | {mv['outcome']} | {mv['missing']} | "
                       f"{'예' if has_r3 else ''} | {label_code((a or {}).get('error_code')) or '없음'} |")
        out.append("")
    return out


def exp_1_7(results, answers, focus_code):
    """특정 코드(기본 R6) 라벨 답안을 모델이 어떻게 판정했는지 + 전체 라벨×모델 교차표."""
    focus_code = focus_code.upper()
    cross = Counter()
    focus = []
    for r in results:
        a = answers.get(r.get("answer_id"))
        if not a:
            continue
        mv = model_view(r, "student")
        lc = label_code(a.get("error_code"))
        mc = "ERR" if mv["error"] else ("기권" if mv["abstain"] else (mv["code"] or "없음"))
        cross[(lc or "없음", mc)] += 1
        if lc == focus_code:
            focus.append((r, mc))
    n_none = sum(1 for _, mc in focus if mc in ("없음", "기권"))
    out = [f"## 1-7 {focus_code} 라벨 답안", "",
           f"라벨이 {focus_code}인 행 {len(focus)}개 중 모델이 코드를 붙이지 않은 행(없음·기권): {_pct(n_none, len(focus))}", ""]
    if focus:
        out += ["| answer_id | 회차 | 모델 판정 |", "| --- | --- | --- |"]
        out += [f"| {r['answer_id']} | {r.get('repeat')} | {mc} |"
                for r, mc in sorted(focus, key=lambda x: (x[0]["answer_id"], str(x[0].get("repeat"))))]
        out.append("")
    labels = sorted({k[0] for k in cross}, key=lambda x: (x == "없음", x))
    preds = sorted({k[1] for k in cross}, key=lambda x: (x in ("없음", "기권", "ERR"), x))
    out += ["### 라벨 × 모델 교차표 (행 수)", "",
            "| 라벨 \\ 모델 | " + " | ".join(preds) + " |", "| --- | " + " | ".join("---" for _ in preds) + " |"]
    for lab in labels:
        out.append(f"| {lab} | " + " | ".join(str(cross.get((lab, p), "")) for p in preds) + " |")
    out.append("")
    return out


def header(results, args):
    meta = Counter((r.get("model"), r.get("prompt_version"), r.get("grader_version"), r.get("codebook_version"))
                   for r in results)
    out = ["# 실험 분석 결과", "",
           f"- 결과 파일: {', '.join(os.path.basename(p) for p in args.results)}",
           f"- 답안 파일: {os.path.basename(args.answers)}",
           f"- 행 수: {len(results)}", "", "| 모델 | 프롬프트 | grader | 코드북 | 행 |", "| --- | --- | --- | --- | --- |"]
    out += [f"| {m[0]} | {m[1]} | {m[2] or '(1.3 이전)'} | {m[3]} | {c} |" for m, c in meta.most_common()]
    if len(meta) > 1:
        out += ["", "> 버전이 섞여 있다. 같은 조건끼리만 비교할 것."]
    out.append("")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--first-col", default="첫_독립_오류", help="라벨의 첫 오류 줄 열 (라벨 통일 후: 첫_독립_오류_기준)")
    ap.add_argument("--first-basis", choices=["student", "criterion"], default="student")
    ap.add_argument("--code-col", default="error_code", help="라벨 주 코드 열 (값 '없음' = 코드 없음)")
    ap.add_argument("--type-col", default="유형", help="오탐검사 여부 열 (값이 '오탐검사'인 답안)")
    ap.add_argument("--focus-ids", default="A-05-B,A-07-B", help="1-3에서 따로 볼 답안 (쉼표 구분, 비우면 생략)")
    ap.add_argument("--focus-code", default="R6", help="1-7에서 볼 라벨 코드")
    ap.add_argument("--include-all", action="store_true", help="holdout '상태'가 유효가 아닌 답안도 포함")
    ap.add_argument("--only", default="", help="일부 실험만: 예) 1-1,1-5")
    ap.add_argument("--out", default="", help="보고서 저장 경로 (.md)")
    args = ap.parse_args(argv)

    answers = load_answers(args.answers, args.include_all)
    missing = [c for c in (args.code_col, args.first_col) if answers and c not in next(iter(answers.values()))]
    if missing:
        raise SystemExit(f"답안 CSV에 열이 없음: {missing}. --code-col / --first-col로 열 이름을 맞출 것")
    for a in answers.values():  # 이하 분석은 error_code / 유형 열 이름을 쓴다
        a["error_code"] = a.get(args.code_col, "")
        a["유형"] = a.get(args.type_col, "")
    results = [r for r in load_results(args.results) if r.get("answer_id") in answers]
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    focus_ids = [x.strip() for x in args.focus_ids.split(",") if x.strip()]

    lines = header(results, args)
    if not only or "1-1" in only:
        lines += exp_1_1(results)
    if not only or only & {"1-2", "1-3"}:
        lines += exp_1_2(results, answers, args.first_basis, args.first_col, focus_ids)
    if not only or "1-5" in only:
        lines += exp_1_5(results, answers)
    if not only or "1-6" in only:
        lines += exp_1_6(results, answers)
    if not only or "1-7" in only:
        lines += exp_1_7(results, answers, args.focus_code)

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n저장: {args.out}", file=sys.stderr)
    return text


if __name__ == "__main__":
    main()
