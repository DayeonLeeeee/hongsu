# coding: utf-8
"""
두 채점 결과를 답안별로 나란히 비교한다.
  - 실험 1-4: 설명문형 답안을 학생 풀이 형태로 바꾼 뒤 판정이 어떻게 달라졌는지
  - 실험 1-1b: 1.3 코드로 재채점한 결과가 11/11 결과(1.2)와 같은지 (--ids all)

변환 전 답안(expert_33)과 변환 후 답안(converted_5)을 각각 run_batch.py로 채점한 결과 JSONL을 받아
같은 answer_id끼리 나란히 놓는다. 판정 정의는 키트 채점기(score_rv2.py)와 같다. LLM을 부르지 않는다.

  python experiments/compare_runs.py \
      --before results/stage3_expert_33_....jsonl --before-answers answers/expert_33.csv \
      --after  results/stage3_converted_5_....jsonl --after-answers answers/converted_5.csv \
      --out experiments/out/1-4_converted.md
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_rv2 as K  # noqa: E402
from score_experiments import answer_vote, group_runs, load_labels, load_results, run_label  # noqa: E402

DEFAULT_IDS = "A-05-D,A-06-A,A-06-C,A-08-A,A-09-B"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--before-answers", required=True)
    ap.add_argument("--after-answers", required=True)
    ap.add_argument("--ids", default=DEFAULT_IDS, help="쉼표 구분 answer_id, 또는 all (양쪽 공통 답안 전부)")
    ap.add_argument("--before-tag", default="변환 전")
    ap.add_argument("--after-tag", default="변환 후")
    ap.add_argument("--title", default="1-4 설명문형 답안 변환 전/후 비교")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    sides = []
    for tag, res, ans in ((args.before_tag, args.before, args.before_answers), (args.after_tag, args.after, args.after_answers)):
        _, labels = load_labels(ans)
        runs = group_runs(load_results(res), labels)
        sides.append((tag, labels, runs))
    if args.ids.strip().lower() == "all":   # 양쪽에 결과가 있는 답안 전부 (재채점 전후 비교용)
        ids = sorted(set(sides[0][2]) & set(sides[1][2]))
    else:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    lines = [f"# {args.title}", "",
             "판정 정의는 키트와 같다 (주 코드 = 반복 다수결, 첫 오류 줄 = matched_criterion).", "",
             "| answer_id | 구분 | 라벨 코드 | 라벨 첫 줄 | AI 회차별 | 다수결 | AI 첫 줄 (회차별) | J (회차별) | 코드 일치 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    changed = both = 0
    for aid in ids:
        majs = []
        for tag, labels, runs in sides:
            lab = labels.get(aid)
            recs = [r for _, r in runs.get(aid, [])]
            if not lab or not recs:
                lines.append(f"| {aid} | {tag} | (라벨 또는 결과 없음) | | | | | | |")
                majs.append(None)
                continue
            maj = answer_vote(recs)
            majs.append(maj)
            firsts = [K.ai_first_error(r["result"]) if r.get("result") else "ERR" for r in recs]
            js = [(r.get("result") or {}).get("justification_level") for r in recs]
            lines.append(f"| {aid} | {tag} | {lab['code'] or '없음'} | {lab['first'] if lab['first'] is not None else '없음'} | "
                         f"{' / '.join(run_label(r) for r in recs)} | {maj} | "
                         f"{' / '.join('없음' if f is None else str(f) for f in firsts)} | {' / '.join(str(j) for j in js)} | "
                         f"{'일치' if K.is_match(maj, lab['code']) else '불일치'} |")
        if len(majs) == 2 and None not in majs:   # 양쪽 결과가 다 있는 답안만 비교
            both += 1
            changed += int(majs[0] != majs[1])
    lines += ["", f"다수결 판정이 달라진 답안: {changed}/{both}"
              + (f" (한쪽 결과가 없는 답안 {len(ids) - both}건 제외)" if both < len(ids) else "")]
    if args.title.startswith("1-4"):
        lines += ["", "변환 후 답안의 라벨은 오세준 교수님 확인 대상 (할 일: '수정한 5건과 통일한 라벨 확인')."]
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return text


if __name__ == "__main__":
    main()
