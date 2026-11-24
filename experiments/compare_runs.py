# coding: utf-8
"""
실험 1-4: 설명문형 답안을 학생 풀이 형태로 바꾼 뒤 판정이 어떻게 달라졌는지 비교.

변환 전 답안과 변환 후 답안을 각각 run_batch.py로 채점한 결과 JSONL 두 개를 받아
같은 answer_id끼리 나란히 놓는다. LLM을 부르지 않는다.

  python experiments/compare_runs.py \
      --before results/stage3_expert_33_....jsonl --before-answers answers/expert_33.csv \
      --after  results/stage3_converted_5_....jsonl --after-answers answers/converted_5.csv \
      --ids A-05-D,A-06-A,A-06-C,A-08-A,A-09-B \
      --out experiments/out/1-4_converted.md
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_experiments import label_code, label_first, load_answers, load_results, model_view  # noqa: E402

DEFAULT_IDS = "A-05-D,A-06-A,A-06-C,A-08-A,A-09-B"


def _fmt(runs):
    if not runs:
        return "(결과 없음)", "", "", ""
    code = " / ".join("ERR" if m["error"] else ("기권" if m["abstain"] else (m["code"] or "없음")) for m in runs)
    first = " / ".join("ERR" if m["error"] else str(m["first"] if m["first"] is not None else "없음") for m in runs)
    j = " / ".join(str(m["J"]) for m in runs)
    outcome = " / ".join(m["outcome"] for m in runs)
    return code, first, j, outcome


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--before-answers", required=True)
    ap.add_argument("--after-answers", required=True)
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--first-col", default="첫_독립_오류_기준")
    ap.add_argument("--first-basis", choices=["student", "criterion"], default="criterion")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    ans_b, ans_a = load_answers(args.before_answers, True), load_answers(args.after_answers, True)

    def by_id(paths):
        d = {}
        for r in load_results(paths):
            d.setdefault(r.get("answer_id"), []).append(r)
        for k in d:
            d[k].sort(key=lambda r: str(r.get("repeat")))
        return d

    rb, ra = by_id(args.before), by_id(args.after)
    lines = ["# 1-4 설명문형 답안 변환 전/후 비교", "",
             f"첫 오류 줄 기준: {'기준표 번호' if args.first_basis == 'criterion' else '학생 줄 번호'} (라벨 열 `{args.first_col}`)",
             "", "| answer_id | 구분 | 라벨 코드 | 라벨 첫 줄 | 모델 코드 | 모델 첫 줄 | J | 결과 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    changed = 0
    for aid in ids:
        row = []
        for tag, res, ans in (("변환 전", rb, ans_b), ("변환 후", ra, ans_a)):
            a = ans.get(aid, {})
            runs = [model_view(r, args.first_basis) for r in res.get(aid, [])]
            code, first, j, outcome = _fmt(runs)
            lc, lf = label_code(a.get("error_code")), label_first(a.get(args.first_col))
            lines.append(f"| {aid} | {tag} | {lc or '없음'} | {lf if lf is not None else '없음'} | {code} | {first} | {j} | {outcome} |")
            row.append((code, first))
        changed += int(row[0] != row[1])
    lines += ["", f"판정(코드·첫 줄)이 달라진 답안: {changed}/{len(ids)}", "",
              "변환 후 답안의 라벨은 오세준 교수님 확인 대상 (할 일: '수정한 5건과 통일한 라벨 확인')."]
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return text


if __name__ == "__main__":
    main()
