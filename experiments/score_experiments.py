# coding: utf-8
"""
미팅 전 필수 실험 분석 (1-1, 1-2, 1-3, 1-5, 1-6, 1-7).

채점 정의는 키트 채점기(experiments/score_rv2.py, 원본 rv2_kit/score_rv2.py)를 그대로 쓴다.
그래서 같은 결과 파일이면 score_rv2.py와 같은 숫자가 나오고, 거기에 실험별 표를 더한다.
결과는 최신 결과 파일로만 본다: 여러 파일을 줄 때는 --latest (또는 * 패턴 + --latest).
LLM을 부르지 않는다 (비용 0).

  python experiments/score_experiments.py \
      --results results/stage3_holdout45_....jsonl --answers answers/holdout45.csv \
      --out experiments/out/holdout_report.md

키트 정의 요약 (score_rv2.py):
  - 답안별 주 코드 = 반복 결과 다수결. 기권·코드누락은 표에서 빼고, 남는 표가 없으면 기권/코드누락.
      없음(오류없음) wrong 줄 없음 / 없음(미완) + response_complete=false
      없음(기권) review_needed=true / 없음(코드누락) wrong 줄은 있는데 코드 없음
  - 라벨이 코드 없음이면 AI가 오류없음·미완일 때만 일치.
  - 첫 오류 줄 = 첫 wrong 줄의 matched_criterion (기준표 번호). 1회차 결과로 비교.
  - holdout: 상태 '유효'만, H-09·H-13 라벨은 R-v3에 따라 R2로 집계.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_rv2 as K  # noqa: E402  (키트 채점기 복사본)

# 정제(형식) 단계에서 나오는 사유. 이 밖의 사유는 검증(내용: 인용·허용 코드·기준 번호 등).
# 지정 답안 확인 기본값: expert = 1-3 모호 답안, holdout = 교수님께 묻기 전에 코드 개선 후 재확인할 답안
#   H-03·H-33: 오류 두 개 — 모델은 앞 줄, 라벨은 뒤 줄 코드 / H-10: 기준표와 오답 유형 불일치 / H-15: R6 정의 범위
DEFAULT_FOCUS = {"expert": "A-05-B,A-07-B", "holdout": "H-03,H-33,H-10,H-15"}

SCHEMA_ISSUE_KEYS = ["결과 블록 없음", "응답이 객체", "lines 없음", "lines가 리스트", "줄이 객체가 아님",
                     "줄 번호(no)", "judgment 값", "math_result_correct 해석", "input_status 값",
                     "justification_level"]


# ─────────────────────────────────────
# 입력
# ─────────────────────────────────────

def resolve_results(paths, latest=False):
    """* 패턴을 펼치고(PowerShell은 패턴을 펼쳐 주지 않음), --latest면 가장 최신 파일 하나만 남긴다.
    최신 = 파일 이름의 실행 시각(…_YYYYMMDD_HHMM) 기준, 같으면 수정 시각."""
    files = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        if not hits:
            raise SystemExit(f"결과 파일 없음: {p}")
        files += hits
    files = list(dict.fromkeys(files))
    if latest and len(files) > 1:
        def key(f):
            m = re.findall(r"(\d{8})_(\d{4})", os.path.basename(f))
            return (m[-1] if m else ("", ""), os.path.getmtime(f))
        files = [max(files, key=key)]
    return files


def load_results(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


CODE_UNIFIED_COL = "전문가_코드_통일"


# 결과 보고서 머리에 버전별로 무엇이 바뀌었는지 적는다 (11/25 결과와 섞어 읽지 않도록).
VERSION_NOTES = {
    "hongsu-rv2-1.3": "출력 형식 규칙 추가(판정 기준 변경 없음). 줄 나누기 규칙 두 문장('학생 답안의 줄 하나 = lines 원소 하나', "
                      "'여러 줄을 한 원소에 합치지 않는다')은 일부러 넣지 않음",
    "hongsu-rv2-1.4": "줄마다 error_criterion(오류가 어긴 기준 번호) 칸 추가 — 11/11 합의 2. matched_criterion은 그대로",
    "hongsu-rv2-1.5": "1.4 + 답안 전체에 stopped_midway(중간에 멈춤) 칸 추가 — 11/11 합의 3. 판정 절차·규칙·코드북·기준표는 1.3과 같음",
    "rv2-grader-1.5": "error_criterion 정수 정리",
    "rv2-grader-1.6": "1.5 + stopped_midway 불리언 정리. 검증 규칙·결과 분류(미완 회차 등)는 1.4와 같음(response_complete 기준)",
}

FIRST_LINE_COL = "전문가_첫오류_학생줄"   # prepare_answers와 같은 이름
ERR_CRIT_COL = "전문가_오류기준"


def load_labels(path, v3_relabel=True, first_col="", code_col="", code_col_optional=False):
    """키트 load_labels + (선택) 라벨 통일본 열로 바꾸기.
    first_col: 첫 오류 줄 (예: 전문가_첫오류_통일)
    code_col:  주 코드 (예: 전문가_코드_통일). 값 'Rk' = 코드, '없음'·빈칸 = 코드 없음, '미완' = 코드 없음 + 미완.
    code_col_optional: 열이 없으면 원래 라벨 그대로 (변환 답안 파일처럼 열이 없을 수 있는 경우)."""
    kind, labels = K.load_labels(path, v3_relabel)
    if first_col or code_col:
        with open(path, encoding="utf-8-sig") as f:
            rows = {r["answer_id"]: r for r in csv.DictReader(f)}
        cols = next(iter(rows.values())).keys() if rows else []
        if first_col:
            if first_col not in cols:
                raise SystemExit(f"답안 CSV에 열이 없음: {first_col}")
            for aid, lab in labels.items():
                lab["first"] = K.to_int(rows.get(aid, {}).get(first_col))
        if code_col and code_col in cols:
            for aid, lab in labels.items():
                v = (rows.get(aid, {}).get(code_col) or "").strip()
                lab["code"] = None if v in ("", "없음", "미완", "코드없음") else v
                if v == "미완":
                    lab["incomplete"] = True
        elif code_col and not code_col_optional:
            raise SystemExit(f"답안 CSV에 열이 없음: {code_col}")
    # 첫 오류 두 열(사람이 채움, prepare_answers). 열이 있고 값이 있는 답안만 1-2b에서 비교한다.
    with open(path, encoding="utf-8-sig") as f:
        rows2 = {r["answer_id"]: r for r in csv.DictReader(f)}
    for aid, lab in labels.items():
        r = rows2.get(aid, {})
        lab["first_line"] = K.to_int(r.get(FIRST_LINE_COL))
        lab["err_crit"] = K.to_int(r.get(ERR_CRIT_COL))
    return kind, labels


def group_runs(results, labels):
    """answer_id → [(repeat, 기록)], 회차 순."""
    runs = defaultdict(list)
    for r in results:
        if r.get("answer_id") in labels:
            runs[r["answer_id"]].append((r.get("repeat"), r))
    for aid in runs:
        runs[aid].sort(key=lambda x: (x[0] is None, x[0] if isinstance(x[0], int) else 0))
    return runs


def run_label(rec):
    """한 회차 → 키트 code_label. 결과가 없으면 ERR."""
    return K.code_label(rec["result"]) if rec.get("result") else "ERR"


def answer_vote(recs):
    labs = [run_label(r) for r in recs if r.get("result")]
    return K.vote(labs) if labs else "ERR"


def _pct(a, b):
    return f"{a}/{b} ({a / b * 100:.1f}%)" if b else f"{a}/0"


def _issue_kind(msg):
    """사유 문장 → (형식|검증, 묶음 이름). 줄 번호와 뒤에 붙은 값은 떼서 같은 종류끼리 센다."""
    for k in SCHEMA_ISSUE_KEYS:
        if k in msg:
            return "형식", k
    m = re.sub(r"^(lines?\s*\[?#?\w*\]?)\s*:\s*", "", msg.strip())
    m = m.split(":", 1)[0].strip()
    m = re.sub(r"\bR\d+\b", "R코드", m)
    return "검증", m[:50]


def _extraction(rec):
    """호출별 결과 출처. 1.3 결과는 extraction 필드, 이전 결과는 응답 원문 블록 종류로 판단."""
    if rec.get("extraction") is not None:
        return list(rec["extraction"])
    out = []
    for raw in rec.get("raw_responses") or []:
        types = [b.get("type") for b in (raw.get("content") or []) if isinstance(b, dict)]
        out.append("tool" if "tool_use" in types else ("text" if "text" in types else "none"))
    return out


# ─────────────────────────────────────
# 실험별
# ─────────────────────────────────────

def exp_1_1(results):
    n = len(results)
    calls = sum(r.get("n_calls") or 0 for r in results)
    ext = Counter(x for r in results for x in _extraction(r))
    retried = sum(1 for r in results if (r.get("n_calls") or 0) > 1)
    errors = sum(1 for r in results if r.get("error"))
    truncated = sum(1 for r in results if "max_tokens" in (r.get("stop_reasons") or []))
    remaining = sum(1 for r in results if r.get("validation_issues"))
    has_first = any("first_issues" in r for r in results)
    kinds, schema_rows = Counter(), 0
    for r in results:
        ks = [_issue_kind(m) for m in (r.get("first_issues") or [])]
        kinds.update(ks)
        schema_rows += any(k[0] == "형식" for k in ks)

    out = ["## 1-1 스키마 안정성", "", "| 항목 | 값 |", "| --- | --- |",
           f"| 실행 행 수 (답안 × 반복) | {n} |",
           f"| 총 API 호출 | {calls} |",
           f"| 결과 출처: 도구 / 텍스트 / 없음 (호출 기준) | {ext.get('tool', 0)} / {ext.get('text', 0)} / {ext.get('none', 0)} |",
           f"| 재시도한 행 (호출 2회) | {_pct(retried, n)} |",
           f"| 재시도 후에도 문제가 남은 행 | {_pct(remaining, n)} |",
           f"| 결과 없음(error) | {_pct(errors, n)} |",
           f"| 출력 잘림(max_tokens) | {_pct(truncated, n)} |"]
    if has_first:
        out.append(f"| 첫 호출에 형식 문제가 있던 행 | {_pct(schema_rows, n)} |")
    out.append("")
    if not has_first:
        out += ["> grader 1.3 이전 결과라 첫 호출 사유(first_issues)는 기록이 없다. "
                "결과 출처는 저장된 응답 원문의 블록 종류로 셌다. 재시도가 0이면 첫 호출이 곧 최종 결과다.", ""]
    if kinds:
        out += ["첫 호출 사유 (형식 = 정제 단계, 검증 = 인용·코드·기준 번호 등 내용 검사)", "",
                "| 구분 | 사유 | 건수 |", "| --- | --- | --- |"]
        out += [f"| {k[0]} | {k[1]} | {c} |" for k, c in kinds.most_common()]
        out.append("")
    return out


def exp_1_2(runs, labels):
    """키트와 같은 첫 오류 줄 일치(1회차) + 회차별 일치 + 답안별 표."""
    first_n = first_d = 0
    rep_hit = defaultdict(lambda: [0, 0])
    table = []
    for aid, lab in labels.items():
        recs = [r for _, r in runs.get(aid, [])]
        if lab["type"] != "오답" or lab["first"] is None or not recs:
            continue
        firsts = [K.ai_first_error(r["result"]) if r.get("result") else "ERR" for r in recs]
        first_d += 1
        first_n += int(firsts[0] == lab["first"])
        for (rep, _), f in zip(runs[aid], firsts):
            rep_hit[rep][0] += int(f == lab["first"])
            rep_hit[rep][1] += 1
        errs = [_first_error_criterion(r) for r in recs]
        table.append((aid, lab, firsts, [run_label(r) for r in recs], errs))

    out = ["## 1-2 첫 오류 줄 일치 (기준표 번호)", "",
           "키트 정의: 라벨 첫 오류(기준표 번호)와 AI 첫 wrong 줄의 matched_criterion을 1회차 결과로 비교.", "",
           f"- **첫 오류 줄 일치: {_pct(first_n, first_d)}**",
           "- 회차별: " + ", ".join(f"{k}회차 {v[0]}/{v[1]}" for k, v in sorted(rep_hit.items(), key=lambda x: str(x[0]))), "",
           "AI 오류 기준(error_criterion)은 프롬프트 1.4부터 나오는 칸으로, 참고용으로 함께 적는다(일치 계산에는 쓰지 않음, 이전 결과는 '-').", "",
           "| answer_id | 라벨 코드 | 라벨 첫 줄 | AI 첫 줄 (회차별) | AI 오류 기준 (회차별) | AI 코드 (회차별) | 일치 회차 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for aid, lab, firsts, codes, errs in sorted(table):
        hit = sum(1 for f in firsts if f == lab["first"])
        out.append(f"| {aid} | {lab['code'] or '없음'} | {lab['first']} | "
                   f"{' / '.join('없음' if f is None else str(f) for f in firsts)} | "
                   f"{' / '.join('-' if e is _NO_FIELD else ('없음' if e is None else str(e)) for e in errs)} | "
                   f"{' / '.join(codes)} | {hit}/{len(firsts)} |")
    out.append("")
    out += _exp_1_2b(runs, labels)
    return out


def _first_wrong(rec):
    lines = [ln for ln in ((rec.get("result") or {}).get("lines") or []) if isinstance(ln, dict)]
    wrong = [ln for ln in lines if ln.get("judgment") == "wrong"]
    return wrong[0] if wrong else None


def _exp_1_2b(runs, labels):
    """첫 오류 두 열 비교: 학생줄(라벨) ↔ AI 첫 wrong 줄의 no, 오류기준(라벨) ↔ AI 첫 wrong 줄의 error_criterion.
    1회차로 비교(1-2와 같은 방식). 라벨 두 열이 비어 있으면 건너뛴다."""
    rows = [(aid, lab) for aid, lab in labels.items()
            if lab.get("first_line") is not None or lab.get("err_crit") is not None]
    if not rows:
        return []
    ln_n = ln_d = ec_n = ec_d = 0
    table = []
    for aid, lab in sorted(rows):
        recs = [r for _, r in runs.get(aid, [])]
        if not recs:
            continue
        fw = [_first_wrong(r) for r in recs]
        nos = [K.to_int(w.get("no")) if w else None for w in fw]
        ecs = [(K.to_int(w.get("error_criterion")) if "error_criterion" in w else _NO_FIELD) if w else None for w in fw]
        if lab.get("first_line") is not None:
            ln_d += 1
            ln_n += int(nos[0] == lab["first_line"])
        if lab.get("err_crit") is not None and ecs[0] is not _NO_FIELD:
            ec_d += 1
            ec_n += int(ecs[0] == lab["err_crit"])
        fmt = lambda xs: " / ".join("-" if x is _NO_FIELD else ("없음" if x is None else str(x)) for x in xs)
        table.append(f"| {aid} | {lab.get('first_line') if lab.get('first_line') is not None else ''} | {fmt(nos)} | "
                     f"{lab.get('err_crit') if lab.get('err_crit') is not None else ''} | {fmt(ecs)} |")
    return ["## 1-2b 첫 오류 두 열 (학생줄 · 오류기준)", "",
            "라벨 '전문가_첫오류_학생줄'·'전문가_오류기준'(사람이 채운 열)과 AI 1회차 첫 wrong 줄의 no·error_criterion을 비교."
            " error_criterion이 없는 결과(프롬프트 1.4 이전)는 오류기준 비교에서 뺀다('-').", "",
            f"- **학생줄 일치: {_pct(ln_n, ln_d)}**",
            f"- **오류기준 일치: {_pct(ec_n, ec_d)}**", "",
            "| answer_id | 라벨 학생줄 | AI 첫 wrong 줄 no (회차별) | 라벨 오류기준 | AI error_criterion (회차별) |",
            "| --- | --- | --- | --- | --- |"] + table + [""]


_NO_FIELD = object()


def _first_error_criterion(rec):
    """첫 wrong 줄의 error_criterion. 1.4 이전 결과처럼 칸이 없으면 _NO_FIELD."""
    lines = [ln for ln in ((rec.get("result") or {}).get("lines") or []) if isinstance(ln, dict)]
    wrong = [ln for ln in lines if ln.get("judgment") == "wrong"]
    if not wrong:
        return None
    if "error_criterion" not in wrong[0]:
        return _NO_FIELD
    return K.to_int(wrong[0].get("error_criterion"))


def _wrong_lines(rec):
    """1회차 wrong 줄들: '학생줄→기준:코드' 목록."""
    lines = [ln for ln in ((rec.get("result") or {}).get("lines") or []) if isinstance(ln, dict)]
    return ", ".join(f"{ln.get('no')}→기준{ln.get('matched_criterion')}"
                     + (f"(오류기준 {ln.get('error_criterion')})" if "error_criterion" in ln else "")
                     + f":{ln.get('code') or '코드없음'}"
                     for ln in lines if ln.get("judgment") == "wrong") or "없음"


def _reason(rec):
    """모델 사유: 검수 사유가 있으면 그것, 없으면 첫 wrong 줄의 reason."""
    res = rec.get("result") or {}
    txt = res.get("review_reason") or ""
    if not txt:
        for ln in res.get("lines") or []:
            if isinstance(ln, dict) and ln.get("judgment") == "wrong" and ln.get("reason"):
                txt = ln["reason"]
                break
    return txt.replace("|", "/").replace("\n", " ")[:150]


def exp_focus(runs, labels, focus_ids, kind):
    """지정 답안 확인 — 1-3(expert 모호 답안)과 '코드 개선 후에도 같은 현상인지' 확인할 holdout 답안."""
    out = ["## 지정 답안 확인 (1-3, 코드 개선 후 재확인)", "",
           "1회차 기준 오류 줄 목록과 모델 사유를 함께 본다. 회차별 코드는 반복 3회.", "",
           "| answer_id | 라벨 코드 | 라벨 첫 줄 | AI 코드 (회차별) | 다수결 | 코드 일치 | AI 오류 줄 (1회차: 학생줄→기준:코드) | 모델 사유 (1회차) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for aid in focus_ids:
        lab = labels.get(aid)
        recs = [r for _, r in runs.get(aid, [])]
        if not lab or not recs:
            out.append(f"| {aid} | (라벨 또는 결과 없음) | | | | | | |")
            continue
        maj = answer_vote(recs)
        first = lab["first"] if lab.get("first") is not None else ("없음" if kind == "expert" else "-")
        out.append(f"| {aid} | {lab['code'] or '없음'} | {first} | {' / '.join(run_label(r) for r in recs)} | {maj} | "
                   f"{'일치' if K.is_match(maj, lab['code']) else '불일치'} | {_wrong_lines(recs[0])} | {_reason(recs[0])} |")
    out.append("")
    return out


def exp_1_5(runs, labels):
    """기권 손실 분류. 키트 '전체 코드 일치'를 기준으로, '코드 없음' 라벨에 대한 기권(·코드누락)을 일치로 볼 때의 수치."""
    base = with_abs = with_abs_miss = 0
    rows = []
    for aid, lab in labels.items():
        recs = [r for _, r in runs.get(aid, [])]
        if not recs:
            continue
        maj = answer_vote(recs)
        per = [run_label(r) for r in recs]
        m = K.is_match(maj, lab["code"])
        base += m
        with_abs += m or (maj == K.NO_ABS and not lab["code"])
        with_abs_miss += m or (maj in (K.NO_ABS, K.NO_MISS) and not lab["code"])
        if K.NO_ABS in per or K.NO_MISS in per:
            reason = ""
            for r in recs:
                if r.get("result") and run_label(r) in (K.NO_ABS, K.NO_MISS):
                    reason = (r["result"].get("review_reason") or "").replace("|", "/").replace("\n", " ")[:90]
                    break
            if maj in (K.NO_ABS, K.NO_MISS):
                cls = "코드 없음 라벨" if not lab["code"] else "진짜 손실"
            else:
                cls = "다수결로 판정됨"
            rows.append((aid, lab["code"] or "없음", per, maj, cls, reason))
    total = sum(1 for aid in labels if runs.get(aid))
    loss = sum(1 for x in rows if x[4] == "진짜 손실")
    out = ["## 1-5 기권 손실 분류", "",
           "답안 단위(반복 다수결). 기준선은 키트 '전체 코드 일치'(코드 없음끼리 일치 포함).", "",
           "| 시나리오 | 일치 |", "| --- | --- |",
           f"| 키트 기준 (기권·코드누락 = 불일치) | {_pct(base, total)} |",
           f"| '코드 없음' 라벨에 대한 기권을 일치로 인정 | {_pct(with_abs, total)} |",
           f"| '코드 없음' 라벨에 대한 기권 + 코드누락을 일치로 인정 | {_pct(with_abs_miss, total)} |", "",
           f"- 기권·코드누락이 한 번이라도 나온 답안: {len(rows)}건",
           f"- 그중 라벨에 코드가 있는데 다수결이 기권·코드누락인 답안 (진짜 손실): **{loss}건**", ""]
    if rows:
        out += ["| answer_id | 라벨 | 회차별 | 다수결 | 분류 | 모델 사유 (첫 기권 회차) |",
                "| --- | --- | --- | --- | --- | --- |"]
        order = {"진짜 손실": 0, "코드 없음 라벨": 1, "다수결로 판정됨": 2}
        for aid, lc, per, maj, cls, reason in sorted(rows, key=lambda x: (order[x[4]], x[0])):
            out.append(f"| {aid} | {lc} | {' / '.join(per)} | {maj} | {cls} | {reason} |")
        out.append("")
    return out


def exp_1_6(runs, labels):
    lab_inc, ai_inc, stop_rows = [], [], []
    for aid, lab in labels.items():
        recs = [r for _, r in runs.get(aid, [])]
        vals = [(r.get("result") or {}).get("response_complete") for r in recs]
        sms = [(r.get("result") or {}).get("stopped_midway", "-") for r in recs]   # 1.5 이전 결과는 '-'
        if lab["incomplete"] or any(x is True for x in sms):
            stop_rows.append((aid, lab, sms))
        r3 = [any((ln.get("code") or "").upper() == "R3" for ln in ((r.get("result") or {}).get("lines") or [])
                  if isinstance(ln, dict)) or ((r.get("result") or {}).get("main_error_code") == "R3") for r in recs]
        if lab["incomplete"]:
            lab_inc.append((aid, vals))
        if any(v is False for v in vals):
            ai_inc.append((aid, lab, vals, [run_label(r) for r in recs],
                           sum(1 for v, x in zip(vals, r3) if v is False and x)))
    n_r3 = sum(x[4] for x in ai_inc)
    out = ["## 1-6 미완 답안", "",
           "| 항목 | 값 |", "| --- | --- |",
           f"| 라벨이 미완인 답안 | {len(lab_inc)} |",
           f"| AI가 한 번이라도 미완으로 본 답안 | {len(ai_inc)} |",
           f"| 미완으로 본 회차에 R3가 붙은 경우 (붙으면 안 됨) | {n_r3} |", ""]
    if lab_inc:
        out += ["라벨 미완 답안의 AI 판단 (회차별 response_complete)", ""]
        out += [f"- {aid}: {vals}" for aid, vals in lab_inc]
        out.append("")
    if ai_inc:
        out += ["| answer_id | 라벨 코드 | 라벨 미완 | AI response_complete (회차별) | AI 코드 (회차별) |",
                "| --- | --- | --- | --- | --- |"]
        for aid, lab, vals, per, _ in sorted(ai_inc, key=lambda x: x[0]):
            out.append(f"| {aid} | {lab['code'] or '없음'} | {'예' if lab['incomplete'] else ''} | {vals} | {' / '.join(per)} |")
        out.append("")
    if any(x != "-" for _, _, sms in stop_rows for x in sms):
        n_lab = sum(1 for _, lab, _ in stop_rows if lab["incomplete"])
        n_hit = sum(1 for _, lab, sms in stop_rows if lab["incomplete"] and sms and sms[0] is True)
        n_extra = sum(1 for _, lab, sms in stop_rows if not lab["incomplete"] and any(x is True for x in sms))
        out += ["### 1-6b 중간에 멈춤 (stopped_midway, 프롬프트 1.5부터)", "",
                "11/11 합의 3의 미완(중간에 멈춘 답안)을 모델이 따로 답한 칸. 위 표의 response_complete(요구 항목 누락)와 다르다."
                " 키트 방식 집계(코드 일치·미완 회차 분류)는 바꾸지 않고 response_complete 기준 그대로 둔다.", "",
                f"- 라벨 미완 답안 중 1회차 stopped_midway=true: {n_hit}/{n_lab}",
                f"- 라벨이 미완이 아닌데 한 번이라도 stopped_midway=true: {n_extra}건", "",
                "| answer_id | 라벨 코드 | 라벨 미완 | AI stopped_midway (회차별) |", "| --- | --- | --- | --- |"]
        out += [f"| {aid} | {lab['code'] or '없음'} | {'예' if lab['incomplete'] else ''} | {sms} |"
                for aid, lab, sms in sorted(stop_rows, key=lambda x: x[0])]
        out.append("")
    return out


def exp_1_7(runs, labels, focus_code):
    focus_code = focus_code.upper()
    focus, cross = [], Counter()
    for aid, lab in labels.items():
        recs = [r for _, r in runs.get(aid, [])]
        if not recs:
            continue
        maj = answer_vote(recs)
        cross[(lab["code"] or "없음", maj)] += 1
        if lab["code"] == focus_code:
            reason = ""
            for r in recs:
                if r.get("result") and not r["result"].get("main_error_code"):
                    reason = (r["result"].get("review_reason") or "").replace("|", "/").replace("\n", " ")[:120]
                    break
            focus.append((aid, [run_label(r) for r in recs], maj, reason))
    n_none = sum(1 for _, _, maj, _ in focus if maj.startswith("없음"))
    out = [f"## 1-7 {focus_code} 라벨 답안", "",
           f"라벨이 {focus_code}인 답안 {len(focus)}건 중 AI 다수결이 코드 없음: **{_pct(n_none, len(focus))}**", ""]
    if focus:
        out += ["| answer_id | AI 회차별 | 다수결 | 코드를 비운 회차의 모델 사유 |", "| --- | --- | --- | --- |"]
        out += [f"| {a} | {' / '.join(p)} | {m} | {r} |" for a, p, m, r in sorted(focus)]
        out.append("")
    labs = sorted({k[0] for k in cross}, key=lambda x: (x == "없음", x))
    preds = sorted({k[1] for k in cross}, key=lambda x: (x.startswith("없음") or x == "ERR", x))
    out += ["### 라벨 × AI(다수결) 교차표 (답안 수)", "",
            "| 라벨 \\ AI | " + " | ".join(preds) + " |", "| --- | " + " | ".join("---" for _ in preds) + " |"]
    for lab in labs:
        out.append(f"| {lab} | " + " | ".join(str(cross.get((lab, p), "")) for p in preds) + " |")
    out.append("")
    return out


def header(results, kind, labels, args):
    meta = Counter((r.get("model"), r.get("prompt_version"), r.get("grader_version"), r.get("codebook_version"))
                   for r in results)
    out = ["# 실험 분석 결과", "",
           f"- 결과 파일: {', '.join(os.path.basename(p) for p in args.results)}",
           f"- 라벨: {os.path.basename(args.answers)} ({kind} {len(labels)}건"
           + (", H-09·H-13은 R2로 집계" if kind == "holdout" and not args.no_v3_relabel else "")
           + (f", 첫 오류 라벨 열 {args.first_col or '전문가_첫오류'}" if kind == "expert" else "")
           + (f", 주 코드 라벨 열 {args.code_col}" if args.code_col else "") + ")",
           f"- 행 수: {len(results)}", "",
           "| 모델 | 프롬프트 | grader | 코드북 | 행 |", "| --- | --- | --- | --- | --- |"]
    out += [f"| {m[0]} | {m[1]} | {m[2] or '(1.3 이전)'} | {m[3]} | {c} |" for m, c in meta.most_common()]
    if len(meta) > 1:
        out += ["", "> 버전이 섞여 있다. 같은 조건끼리만 비교할 것."]
    notes = [f"- **{k}**: {v}" for k, v in VERSION_NOTES.items()
             if any(k in str(m[1]) or k == m[2] for m in meta)]
    if notes:
        out += ["", "이 결과의 버전에서 바뀐 점 (CHANGELOG.md):"] + notes
    out.append("")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--answers", required=True, help="키트 형식 라벨 CSV (expert_33.csv 또는 holdout45.csv)")
    ap.add_argument("--first-col", default="", help="첫 오류 라벨을 다른 열에서 읽기 (예: 라벨 통일본 열)")
    ap.add_argument("--code-col", default="", help="주 코드 라벨을 다른 열에서 읽기 (예: 전문가_코드_통일, '미완' = 코드 없음)")
    ap.add_argument("--focus-ids", default=None,
                    help="지정 답안 확인에서 볼 답안 (쉼표 구분, 빈 문자열이면 생략). "
                         f"기본: expert {DEFAULT_FOCUS['expert']} / holdout {DEFAULT_FOCUS['holdout']}")
    ap.add_argument("--latest", action="store_true",
                    help="--results가 여러 파일(또는 * 패턴)일 때 가장 최신 파일 하나만 쓴다")
    ap.add_argument("--focus-code", default="R6", help="1-7에서 볼 라벨 코드")
    ap.add_argument("--no-v3-relabel", action="store_true", help="holdout H-09·H-13 R2 재라벨을 하지 않음")
    ap.add_argument("--only", default="", help="일부 실험만: 예) 1-1,1-5")
    ap.add_argument("--out", default="", help="보고서 저장 경로 (.md)")
    args = ap.parse_args(argv)

    args.results = resolve_results(args.results, args.latest)
    kind, labels = load_labels(args.answers, not args.no_v3_relabel, args.first_col, args.code_col)
    if kind == "holdout" and args.no_v3_relabel:
        for aid in K.V3_RELABEL:
            labels.pop(aid, None)
    results = [r for r in load_results(args.results) if r.get("answer_id") in labels]
    runs = group_runs(results, labels)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    focus_raw = DEFAULT_FOCUS.get(kind, "") if args.focus_ids is None else args.focus_ids
    focus_ids = [x.strip() for x in focus_raw.split(",") if x.strip()]

    lines = header(results, kind, labels, args)
    if not only or "1-1" in only:
        lines += exp_1_1(results)
    if kind == "expert" and (not only or "1-2" in only):
        lines += exp_1_2(runs, labels)
    if focus_ids and (not only or only & {"1-3", "focus"}):
        lines += exp_focus(runs, labels, focus_ids, kind)
    if not only or "1-5" in only:
        lines += exp_1_5(runs, labels)
    if not only or "1-6" in only:
        lines += exp_1_6(runs, labels)
    if not only or "1-7" in only:
        lines += exp_1_7(runs, labels, args.focus_code)

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
