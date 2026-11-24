# -*- coding: utf-8 -*-
# [복사본] 원본: rv2_kit/score_rv2.py (2026-11-11 수정본). 팀이 교수님께 보고한 수치의 채점 기준이다.
#          정의를 바꾸려면 키트 원본과 함께 바꿀 것. experiments/score_experiments.py가 이 파일의 함수를 쓴다.
"""
score_rv2.py — Stage 3 채점 결과(JSONL)를 전문가/확정 라벨과 대조한다. (키트 score.py의 서버용 개정판)

입력 결과: run_batch.py 출력 또는 키트 run_eval.py 출력 (둘 다 줄마다 answer_id·result를 가진 JSONL)
입력 라벨: expert_33.csv 형식 또는 holdout45 확정라벨 CSV 형식 (열 이름으로 자동 판별)

키트 score.py 대비 바뀐 점 (11/8 미팅 후속):
  [1] 첫 오류 줄 = '기준 판정표 번호'로 비교한다.
      전문가는 기준 번호로 라벨했으므로, AI 쪽도 학생 줄 번호(no) 대신 matched_criterion을 쓴다.
      AI 첫 오류 = 첫 wrong 줄의 matched_criterion.
                   wrong 줄이 없는데 main_error_code가 있으면(= 안 쓴 오류, 예: R3) missing_criteria의 첫 번호.
  [탐지] 오답 탐지율 분모에서, 유형은 '오답'이지만 전문가가 오류 없다고 본 답안(코드 없음 + 미완 아님)을 뺀다.
      11/8 미팅 질문 8-1에 대한 교수님 답변("분모에서 제외하는 게 맞다") 반영.
        expert_33: A-09-B(전부 ok)·A-10-B(결론 맞음, 반례 없음) 제외 / A-08-A(미완)는 탐지 대상으로 유지
        holdout  : 코드없음 라벨 중 비고에 미완·누락이 적힌 H-43·H-37은 유지, 나머지 제외
      --detect-keep-all 을 주면 키트와 같이 유형 '오답' 전부를 분모로 쓴다.
      '탐지'에는 main_error_code만 있는 경우(안 쓴 오류, wrong 줄 없이 코드만 부여)도 포함한다.
  [+] 미완 기록: 라벨 비고에 미완(response_complete=false)이 적힌 답안에서 AI가 response_complete=false를 냈는지.
  [+] holdout: 상태 '유효'만 집계, 코드북 R-v3 확정에 따라 H-09·H-13 라벨을 R2로 바꿔 집계(--no-v3-relabel로 끔).
      안내문 기준 성능(main 35/38)과 비교할 수 있게 '전체 코드 일치(코드없음끼리 일치 포함)'도 출력한다.

  [+] '코드 없음'을 이유별로 나눠 표시한다 (결과 필드로 판단, 프롬프트 변경 없음):
        없음(오류없음)  wrong 줄 없음 — 정답으로 판정
        없음(미완)      wrong 줄 없음 + response_complete=false
        없음(기권)      review_needed=true — 맞는 코드가 없어 판단 보류
        없음(코드누락)  wrong 줄은 있는데 코드가 없음 — 출력 이상
      다수결: 오류없음·미완은 한 표, 기권·코드누락은 표에서 뺀다.
              전 회차가 기권/코드누락이면 그대로 '없음(기권)'/'없음(코드누락)'이 되고, 이는 불일치로 센다.
      라벨이 코드 없음이면 AI가 없음(오류없음)·없음(미완)일 때만 일치.

판정 방식: main 코드는 위 다수결, 첫 오류 줄·J·오탐 줄은 1회차 결과(키트와 동일), 불안정은 main_error_code 기준.

사용:
  python score_rv2.py --results results/stage3_expert_33_....jsonl --answers answers/expert_33.csv
  python score_rv2.py --results results/stage3_holdout45_....jsonl --answers answers/holdout45.csv
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

CIRCLED = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩")}
V3_RELABEL = {"H-09": "R2", "H-13": "R2"}   # 코드북 R-v3 개정안 1번 (R2 재정의) 확정 사항


def to_int(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s and s[0] in CIRCLED:
        return CIRCLED[s[0]]
    m = re.fullmatch(r"(?:기준)?\s*\(?\s*(\d+)\s*\)?", s)   # "2", "기준2", "(2)" 모두 허용
    return int(m.group(1)) if m else None


NO_OK, NO_INC, NO_ABS, NO_MISS = "없음(오류없음)", "없음(미완)", "없음(기권)", "없음(코드누락)"
NON_VOTE = (NO_ABS, NO_MISS)


def majority(values):
    c = Counter(values).most_common()
    if not c:
        return None, 0.0
    return c[0][0], c[0][1] / len(values)


def code_label(res: dict) -> str:
    """한 회차 결과 → 'Rk' 또는 '없음(이유)'."""
    if res.get("main_error_code"):
        return res["main_error_code"]
    if res.get("review_needed") is True:
        return NO_ABS
    if any(isinstance(ln, dict) and ln.get("judgment") == "wrong" for ln in (res.get("lines") or [])):
        return NO_MISS
    if res.get("response_complete") is False:
        return NO_INC
    return NO_OK


def vote(labels: list) -> str:
    """기권·코드누락은 표에서 빼고 다수결. 남는 표가 없으면 기권(하나라도 있으면) 또는 코드누락."""
    votes = [l for l in labels if l not in NON_VOTE]
    if votes:
        return majority(votes)[0]
    return NO_ABS if NO_ABS in labels else NO_MISS


def is_match(ai_label: str, label_code) -> bool:
    if label_code:
        return ai_label == label_code
    return ai_label in (NO_OK, NO_INC)


def load_labels(path: str, v3_relabel: bool) -> tuple:
    """→ (형식, {answer_id: label}). label = type('오답'|'정답류'), code(None|'Rk'), first(int|None),
    j, incomplete(bool), note"""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = {}
    if rows and "확정_주" in rows[0]:
        kind = "holdout"
        for r in rows:
            if r["상태"].strip() != "유효":
                continue
            main = r["확정_주"].strip()
            code = None if main in ("정답", "코드없음", "") else main
            if v3_relabel and r["answer_id"] in V3_RELABEL:
                code = V3_RELABEL[r["answer_id"]]
            note = r.get("조정·비고", "") or ""
            labels[r["answer_id"]] = {
                "problem_id": r["problem_id"], "type": "정답류" if main == "정답" else "오답",
                "code": code, "first": None, "j": r["확정_J"].strip(),
                "incomplete": ("response_complete=false" in note) or ("누락" in note), "note": note}
    else:
        kind = "expert"
        for r in rows:
            raw = (r.get("전문가_코드") or "").strip()
            note = r.get("비고", "") or ""
            labels[r["answer_id"]] = {
                "problem_id": r["problem_id"], "type": "오답" if r.get("유형", "") == "오답" else "정답류",
                "code": None if raw in ("", "없음") else raw,
                "first": to_int(r.get("전문가_첫오류")), "j": (r.get("전문가_J") or "").strip(),
                "incomplete": ("response_complete=false" in note) or ("미완" in note)
                              or ("누락" in (r.get("전문가_첫오류") or "")), "note": note}
    return kind, labels


def ai_first_error(res: dict):
    """[1] AI 첫 오류 = 첫 wrong 줄(학생 줄 순서)의 matched_criterion.
    wrong 줄이 없는데 main_error_code가 있으면 missing_criteria의 첫 번호 (안 쓴 오류)."""
    lines = [ln for ln in (res.get("lines") or []) if isinstance(ln, dict)]
    wrongs = sorted((ln for ln in lines if ln.get("judgment") == "wrong"),
                    key=lambda ln: to_int(ln.get("no")) or 0)
    if wrongs:
        return to_int(wrongs[0].get("matched_criterion"))
    missing = sorted(n for n in (to_int(x) for x in (res.get("missing_criteria") or [])) if n)
    if res.get("main_error_code") and missing:
        return missing[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--detect-keep-all", action="store_true",
                    help="오답 탐지율 분모를 키트와 같이 유형 '오답' 전부로 계산 (기본: 오류 없음 라벨은 제외)")
    ap.add_argument("--no-v3-relabel", action="store_true",
                    help="holdout H-09·H-13을 R2로 바꾸지 않고 집계에서 제외 (안내문 기준 성능 38건과 동일 조건)")
    args = ap.parse_args()

    kind, labels = load_labels(args.answers, not args.no_v3_relabel)
    if kind == "holdout" and args.no_v3_relabel:
        for aid in V3_RELABEL:
            labels.pop(aid, None)

    runs = defaultdict(list)
    meta = Counter()
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("result"):
                runs[r["answer_id"]].append(r["result"])
                meta[(r.get("model"), r.get("prompt_version"), r.get("codebook_version"))] += 1

    stat = Counter()
    confusion = Counter()
    false_pos, unstable, missing_runs, incomplete_rows, rows = [], [], [], [], []
    abstain_rows, codeless_rows = [], []

    for aid, lab in labels.items():
        results = runs.get(aid)
        if not results:
            missing_runs.append(aid)
            continue
        per_run = [code_label(res) for res in results]
        maj = vote(per_run)
        maj_code = maj if maj and not maj.startswith("없음") else None      # 실제 R코드만
        _, agree = majority([res.get("main_error_code") for res in results])
        if len(results) > 1 and agree < 1.0:
            unstable.append(aid)
        if NO_ABS in per_run:
            abstain_rows.append((aid, lab["code"] or "없음", per_run, maj))
        if NO_MISS in per_run:
            codeless_rows.append((aid, per_run))
        res = results[0]
        wrong_lines = [ln for ln in (res.get("lines") or []) if isinstance(ln, dict) and ln.get("judgment") == "wrong"]
        detected = (res.get("math_result_correct") is False) or bool(wrong_lines) or bool(res.get("main_error_code"))
        ai_first = ai_first_error(res)
        ai_j = res.get("justification_level")

        # 전체 코드 일치 (코드 없음끼리도 일치로 셈) — holdout 안내문 기준 성능과 같은 정의
        stat["all_d"] += 1
        stat["all_n"] += int(is_match(maj, lab["code"]))

        if lab["type"] == "오답" and (args.detect_keep_all or lab["code"] or lab["incomplete"]):  # 탐지율 분모
            stat["det_d"] += 1
            stat["det_n"] += int(detected)
        if lab["type"] == "오답" and lab["code"]:          # 코드 일치 분모: 라벨에 코드가 있는 답안
            stat["code_d"] += 1
            stat["code_n"] += int(is_match(maj, lab["code"]))
            confusion[(lab["code"], maj)] += 1
        if lab["type"] == "오답" and lab["first"] is not None:   # [1] 기준 번호끼리 비교
            stat["first_d"] += 1
            stat["first_n"] += int(ai_first == lab["first"])
        if lab["type"] == "정답류" and (maj_code or wrong_lines):
            false_pos.append(aid)
        if lab["j"] and lab["j"] != "없음":
            stat["j_d"] += 1
            stat["j_n"] += int(ai_j == lab["j"])
        if lab["incomplete"]:
            incomplete_rows.append((aid, [r_.get("response_complete") for r_ in results]))

        rows.append([aid, lab["problem_id"], lab["type"], lab["code"] or "없음", maj, " / ".join(per_run),
                     lab["first"] if lab["first"] is not None else "", ai_first if ai_first is not None else "",
                     lab["j"], ai_j, res.get("response_complete"),
                     json.dumps(res.get("missing_criteria") or []), int(is_match(maj, lab["code"]))])

    def pct(n, d):
        return f"{n}/{d} ({n / d * 100:.1f}%)" if d else "n/a"

    n_correct_type = sum(1 for l in labels.values() if l["type"] == "정답류")
    print("=" * 60)
    print(f"라벨: {kind} {len(labels)}건 / 결과 있는 답안 {len(rows)}건"
          + (f" / 결과 없음 {missing_runs}" if missing_runs else ""))
    for (m, pv, cv), c in meta.items():
        print(f"  기록: model={m} prompt_version={pv} codebook_version={cv} ({c}회)")
    if kind == "holdout":
        print("  H-09·H-13: " + ("집계 제외 (--no-v3-relabel)" if args.no_v3_relabel else "R-v3 확정에 따라 R2로 집계"))
    print("-" * 60)
    print(f"오답 탐지율              : {pct(stat['det_n'], stat['det_d'])}"
          + ("  (유형 '오답' 전부)" if args.detect_keep_all else "  (오류 없음 라벨 제외, 미완은 포함)"))
    if kind == "expert":
        print(f"첫 오류 줄 일치 [1]      : {pct(stat['first_n'], stat['first_d'])}  (기준 판정표 번호끼리 비교)")
    print(f"main_error_code 일치     : {pct(stat['code_n'], stat['code_d'])}  (라벨에 코드가 있는 답안)")
    print(f"전체 코드 일치           : {pct(stat['all_n'], stat['all_d'])}  (코드없음끼리 일치 포함)")
    print(f"J 수준 일치              : {pct(stat['j_n'], stat['j_d'])}")
    print(f"오탐 (정답류 {n_correct_type}건에 코드/wrong): {len(false_pos)}건 {false_pos if false_pos else '- 통과'}")
    print(f"반복 간 불안정 답안      : {len(unstable)}건 {unstable}")
    print(f"기권 포함 답안          : {len(abstain_rows)}건"
          + ("" if not abstain_rows else "  (전 회차 기권 → 불일치로 셈)"))
    for aid, lc, pr, m in abstain_rows:
        print(f"  {aid} [라벨 {lc}] 회차별: {' / '.join(pr)} → 판정 {m}"
              + ("  ← 전 회차 기권" if m == NO_ABS else ""))
    if codeless_rows:
        print(f"코드누락(wrong인데 코드 없음): {len(codeless_rows)}건")
        for aid, pr in codeless_rows:
            print(f"  {aid} 회차별: {' / '.join(pr)}")
    if incomplete_rows:
        print("미완 기록 (라벨: response_complete=false → AI 값, 회차별):")
        for aid, vals in incomplete_rows:
            print(f"  {aid}: {vals}")
    if confusion:
        print("[혼동표] 라벨 → AI(다수결) : 건수")
        for (e, a), c in sorted(confusion.items()):
            print(f"  {e:>4} → {a:<8}: {c}" + ("" if e == a else "  <-- 불일치"))
    print("=" * 60)

    base = os.path.splitext(os.path.basename(args.results))[0]
    out_csv = os.path.join(os.path.dirname(os.path.abspath(args.results)), f"score_{base}.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["answer_id", "problem_id", "유형", "라벨_코드", "AI_코드(다수결)", "AI_회차별", "라벨_첫오류(기준)",
                    "AI_첫오류(기준)", "라벨_J", "AI_J", "AI_response_complete", "AI_missing_criteria", "코드일치"])
        w.writerows(rows)
    print(f"상세표 저장: {out_csv}")


if __name__ == "__main__":
    main()
