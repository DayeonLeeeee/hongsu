# -*- coding: utf-8 -*-
"""
run_batch.py — 답안 CSV를 hongsu 진단 로직(Stage 3, with_codebook)으로 일괄 채점해 JSONL로 저장한다.
채점 결과의 일치율은 score_rv2.py로 계산한다.

두 가지 방식:
  (기본) 로컬 실행 — 서버와 같은 코드(rv2_grader.diagnose, 엑셀 materials/dataset_rv2.xlsx)를 직접 호출.
  --url  배포 서버 실행 — POST {url}/grade 로 보낸다 (서버 검증 보고용. 느림).

사용:
  python run_batch.py --answers answers/expert_33.csv --repeats 3
  python run_batch.py --answers answers/holdout45.csv --repeats 3 --url https://<hongsu 주소>

  * holdout CSV는 상태가 '유효'인 40건만 채점한다 (--include-all 이면 45건 전부).
  * 모델은 환경변수 RV2_MODEL (기본 claude-haiku-4-5).
  * 기록마다 model / prompt_version / codebook_version / xlsx_sha8 이 남는다 (holdout 사용 규칙 3).
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_answers(path: str, include_all: bool) -> list:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and "상태" in rows[0] and not include_all:
        rows = [r for r in rows if r["상태"].strip() == "유효"]
    return rows


def grade_local(client, row: dict) -> dict:
    import rv2_grader
    rec = rv2_grader.diagnose(client, row["problem_id"], row["학생_답안"], mode="with_codebook")
    rec.pop("answer_text", None)
    return rec


def grade_remote(url: str, row: dict) -> dict:
    import requests
    resp = requests.post(url.rstrip("/") + "/grade", timeout=300, json={
        "problem_id": row["problem_id"], "solution_text": row["학생_답안"],
        "mode": "with_codebook", "student_id": f"batch:{row['answer_id']}",
    })
    d = resp.json()
    result_keys = ["lines", "main_error_code", "justification_level", "math_result_correct",
                   "input_status", "review_needed", "review_reason", "response_complete", "missing_criteria"]
    return {
        "model": d.get("model"), "prompt_version": d.get("prompt_version"),
        "codebook_version": d.get("codebook_version"), "xlsx_sha8": d.get("xlsx_sha8"),
        "allowed_codes": d.get("allowed_codes"), "usage": d.get("usage"), "n_calls": d.get("n_calls"),
        "stop_reasons": d.get("stop_reasons"), "validation_issues": d.get("validation_issues"),
        "error": d.get("error") or (None if d.get("success") else f"HTTP {resp.status_code}"),
        "server_answer_id": d.get("answer_id"),
        "result": {k: d.get(k) for k in result_keys} if d.get("lines") is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--url", default="", help="배포 서버 주소. 비우면 로컬 실행")
    ap.add_argument("--include-all", action="store_true", help="holdout의 '1차심사 미달'까지 채점")
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "results"))
    args = ap.parse_args()

    rows = load_answers(args.answers, args.include_all)
    client = None
    if not args.url:
        sys.path.insert(0, BASE_DIR)
        import anthropic
        import rv2_codebook
        import rv2_grader
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        model, cbv = rv2_grader.RV2_MODEL, rv2_codebook.get().get("codebook_version")
    else:
        model, cbv = "server", "server"

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.answers))[0]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(args.out_dir, f"stage3_{stem}_{model}_{cbv}_{ts}.jsonl")
    print(f"{len(rows)}건 x {args.repeats}회 → {out_path}")

    with open(out_path, "w", encoding="utf-8") as out:
        for rep in range(1, args.repeats + 1):
            for row in rows:
                try:
                    rec = grade_remote(args.url, row) if args.url else grade_local(client, row)
                except Exception as e:
                    rec = {"result": None, "error": f"{type(e).__name__}: {e}"}
                rec.update({"answer_id": row["answer_id"], "problem_id": row["problem_id"], "repeat": rep})
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                res = rec.get("result") or {}
                print(f"  [{rep}/{args.repeats}] {row['answer_id']}: "
                      + (f"ERR {rec['error'][:60]}" if rec.get("error") else
                         f"main={res.get('main_error_code')} J={res.get('justification_level')} "
                         f"complete={res.get('response_complete')} missing={res.get('missing_criteria')}"))
                time.sleep(0.3)
    print(f"완료. 채점: python score_rv2.py --results {out_path} --answers {args.answers}")


if __name__ == "__main__":
    main()
