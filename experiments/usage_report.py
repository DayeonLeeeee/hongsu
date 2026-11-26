# coding: utf-8
"""
채점 결과 JSONL의 API 사용량과 비용 (LLM 호출 없음).

  python experiments/usage_report.py results/stage3_expert_33_*.jsonl results/stage3_holdout45_*.jsonl

파일마다: 기록 수, 호출 수(재시도 포함), 입력·출력 토큰, 비용 합계, 답안 1건당 비용.
비용은 기록의 cost_usd(1.3 이후), 없으면 응답 원문의 호출별 모델·토큰으로 다시 계산한다 (usage_cost.PRICES).
예전 결과도 같은 가격표로 계산하므로 '지금 가격으로 다시 돌리면 드는 돈'이다.
"""
import argparse
import glob
import json
import os
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import usage_cost  # noqa: E402


def report(paths: List[str]) -> str:
    lines = [f"# API 사용량 (가격표 {usage_cost.PRICES_CHECKED} 기준, USD)", "",
             "| 파일 | 기록 | 호출 | 입력 토큰 | 출력 토큰 | 비용 | 1건당 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    total = 0.0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = usage_cost.summarize(json.loads(line) for line in f if line.strip())
        total += s["cost_usd"]
        per = f"${s['cost_per_record_usd']:.4f}" if s["cost_per_record_usd"] is not None else "?"
        note = f" (가격 모름 {s['unpriced_records']}건)" if s["unpriced_records"] else ""
        lines.append(f"| {os.path.basename(p)} | {s['records']} | {s['calls']} | {s['input_tokens']:,} | "
                     f"{s['output_tokens']:,} | ${s['cost_usd']:.2f}{note} | {per} |")
    lines += ["", f"합계: ${total:.2f}"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="결과 JSONL (glob 가능)")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    paths = sorted({p for pat in a.results for p in (glob.glob(pat) or [pat])})
    text = report(paths)
    print(text)
    return text


if __name__ == "__main__":
    main()
