# coding: utf-8
"""
실험 3-3 프롬프트 회귀 테스트 — 고정 답안 20개를 프롬프트·채점 코드를 바꿀 때마다 돌려 기준과 비교한다.

  python experiments/regression.py --save-baseline   현재 코드로 채점해 기준으로 저장 (처음 한 번, 의도한 변경 뒤)
  python experiments/regression.py                   현재 코드로 채점해 기준과 비교 (API 20회)
  python experiments/regression.py --compare <run.jsonl>   이미 돌린 결과만 다시 비교 (비용 0)

- 답안: materials/dataset_rv2.xlsx의 04_SYNTHETIC_ANSWERS (expert_33과 같은 원본). 키트 없이 돈다.
  holdout은 쓰지 않는다 (평가용 데이터를 개발 반복에 쓰면 오염된다).
- 고정 세트 20건: 문항 10개마다 오답 1건(A-xx-A) + 오탐검사 1건 → 모든 문항, 오답·정답류를 다 덮는다.
- 비교 항목: 주 코드, 첫 오류 줄(matched_criterion), J, outcome, 미완 여부. 하나라도 다르면 '바뀜'.
- 결과: experiments/regression/run_<시각>.jsonl, 보고서 experiments/regression/report_<시각>.md
  기준: experiments/regression/baseline.jsonl (저장소에 올린다. 합성 답안이라 공개해도 된다)
- 반복 1회: Opus 5.5는 11/11 실험에서 반복 간 변동이 0건이었다. 바뀐 건은 다시 돌려 우연인지 확인한다.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

REG_DIR = os.path.join(HERE, "regression")
BASELINE = os.path.join(REG_DIR, "baseline.jsonl")
REGRESSION_IDS = [
    "A-01-A", "A-01-C", "A-02-A", "A-02-C", "A-03-A", "A-03-C", "A-04-A", "A-04-C", "A-05-A", "A-05-D",
    "A-06-A", "A-06-C", "A-07-A", "A-07-C", "A-08-A", "A-08-D", "A-09-A", "A-09-C", "A-10-A", "A-10-D",
]
FIELDS = ["main_error_code", "first_error", "justification_level", "outcome", "response_complete"]


def load_rows(ids: List[str] = REGRESSION_IDS) -> List[dict]:
    """엑셀 04_SYNTHETIC_ANSWERS에서 고정 세트 답안을 순서대로."""
    import prepare_answers
    import rv2_codebook
    rows = {r["answer_id"]: r for r in prepare_answers.expert_from_xlsx(rv2_codebook.XLSX_PATH)}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"엑셀에 없는 답안: {missing}")
    return [rows[i] for i in ids]


def summarize(rec: dict) -> Dict[str, object]:
    """채점 기록 1건 → 비교 항목."""
    import score_rv2 as K
    res = rec.get("result") or {}
    return {
        "main_error_code": res.get("main_error_code"),
        "first_error": K.ai_first_error(res) if res else None,
        "justification_level": res.get("justification_level"),
        "outcome": rec.get("outcome"),
        "response_complete": res.get("response_complete"),
    }


def load_jsonl(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["answer_id"]: r for r in (json.loads(line) for line in f if line.strip())}


def grade(rows: List[dict], out_path: str, client=None) -> str:
    import rv2_grader
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                                     max_retries=int(os.environ.get("CLAUDE_MAX_RETRIES", "2")),
                                     timeout=float(os.environ.get("CLAUDE_TIMEOUT_S", "120")))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            rec = rv2_grader.diagnose(client, row["problem_id"], row["학생_답안"], mode="with_codebook")
            rec["outcome"], rec["review_sources"] = rv2_grader.classify_outcome(rec)
            rec.update({"answer_id": row["answer_id"], "repeat": 1})
            rec.pop("answer_text", None)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            s = summarize(rec)
            print(f"  [{i}/{len(rows)}] {row['answer_id']}: " + (f"ERR {rec['error'][:60]}" if rec.get("error") else
                  f"main={s['main_error_code']} first={s['first_error']} J={s['justification_level']} "
                  f"outcome={s['outcome']}"), flush=True)
    return out_path


def compare(run_path: str, baseline_path: str = BASELINE, ids: List[str] = REGRESSION_IDS) -> dict:
    """→ {'changed': [...], 'missing': [...], 'table': markdown, 'versions': (기준, 이번)}"""
    base, run = load_jsonl(baseline_path), load_jsonl(run_path)
    lines = ["| answer_id | 항목 | 기준 | 이번 |", "| --- | --- | --- | --- |"]
    changed, missing = [], []
    for aid in ids:
        if aid not in base or aid not in run:
            missing.append(aid)
            continue
        b, r = summarize(base[aid]), summarize(run[aid])
        diff = [k for k in FIELDS if b[k] != r[k]]
        if diff:
            changed.append(aid)
            for k in diff:
                lines.append(f"| {aid} | {k} | {b[k]} | {r[k]} |")
    def ver(recs: Dict[str, dict]) -> str:
        r = next(iter(recs.values()), {})
        return f"{r.get('prompt_version')} / {r.get('grader_version')} / {r.get('model')}"
    return {"changed": changed, "missing": missing, "table": "\n".join(lines),
            "versions": (ver(base), ver(run))}


def report(cmp: dict, run_path: str, n: int = len(REGRESSION_IDS)) -> str:
    out = ["# 프롬프트 회귀 테스트 (실험 3-3)", "",
           f"- 기준: {cmp['versions'][0]}", f"- 이번: {cmp['versions'][1]}",
           f"- 결과 파일: {os.path.relpath(run_path, ROOT)}", "",
           f"**바뀐 답안: {len(cmp['changed'])}/{n - len(cmp['missing'])}**"
           + (f" (결과 없음 {len(cmp['missing'])}건: {', '.join(cmp['missing'])})" if cmp["missing"] else ""), ""]
    if cmp["changed"]:
        out += [cmp["table"], "",
                "바뀐 답안은 한 번 더 돌려 같은 결과인지 확인한다. 의도한 변경이면 `--save-baseline`으로 기준을 갱신하고, "
                "그 사실을 커밋 메시지에 남긴다."]
    else:
        out.append("기준과 같다.")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-baseline", action="store_true", help="채점 결과를 기준으로 저장")
    ap.add_argument("--compare", default="", help="이미 있는 결과 파일만 비교 (API 호출 없음)")
    ap.add_argument("--baseline", default=BASELINE)
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if a.compare:
        run_path = a.compare
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY가 없음 (채점 20회 호출)")
        run_path = os.path.join(REG_DIR, f"run_{ts}.jsonl")
        print(f"고정 답안 {len(REGRESSION_IDS)}건 채점 → {os.path.relpath(run_path, ROOT)}")
        grade(load_rows(), run_path)

    if a.save_baseline:
        shutil.copyfile(run_path, a.baseline)
        print(f"기준 저장: {os.path.relpath(a.baseline, ROOT)}")
        return 0
    if not os.path.exists(a.baseline):
        raise SystemExit("기준이 없음. 먼저: python experiments/regression.py --save-baseline")
    cmp = compare(run_path, a.baseline)
    text = report(cmp, run_path)
    print("\n" + text)
    with open(os.path.join(REG_DIR, f"report_{ts}.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 1 if cmp["changed"] or cmp["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
