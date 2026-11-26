# coding: utf-8
"""API 사용량 → 비용(USD) 환산, 그리고 서버가 켜진 뒤의 누적 사용량.

가격 (2026-11-25 확인, 표준 요금):
  Claude  https://platform.claude.com/docs/en/about-claude/pricing (global, Batch 아님)
  Gemini  https://ai.google.dev/gemini-api/docs/pricing (유료 Standard. 출력에는 thinking 토큰 포함.
          무료 등급 키를 쓰면 실제 청구는 0이다)
  Mathpix https://mathpix.com/pricing/api (이미지 OCR v3/text, 요청 1건당. 100만 건까지 $0.002)
  단위 USD / 백만 토큰. cache_write는 5분 캐시 쓰기 요금 (hongsu는 캐시를 쓰지 않아 보통 0).
  가격이 바뀌면 PRICES와 PRICES_CHECKED를 같이 고친다. 표에 없는 모델은 비용을 None으로 둔다 (추정하지 않음).
  Haiku 5.5는 프롬프트 10만 토큰까지 요금. 채점 프롬프트는 그보다 훨씬 짧다.

채점 서버(/grade)·실험(run_batch, regression)·OCR 수정(/refine)이 같은 함수로 계산한다.
"""
import threading
from collections import defaultdict
from typing import Dict, Iterable, Optional

PRICES_CHECKED = "2026-11-25"
PRICES: Dict[str, Dict[str, float]] = {
    "claude-opus-5-5":   {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
    "claude-haiku-5-5":  {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-opus-4-6":   {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "gemini-2.5-flash":  {"input": 0.30, "output": 2.50, "cache_write": 0.0, "cache_read": 0.0},
}
# 토큰이 아니라 요청 1건마다 과금하는 서비스
PER_REQUEST: Dict[str, float] = {"mathpix": 0.002}
_FIELDS = {"input_tokens": "input", "output_tokens": "output",
           "cache_creation_input_tokens": "cache_write", "cache_read_input_tokens": "cache_read"}


def price_for(model: Optional[str]) -> Optional[Dict[str, float]]:
    """모델 ID(날짜 접미사 포함, 예: claude-haiku-4-5-20251001)의 가격. 모르면 None."""
    if not model:
        return None
    for k, v in PRICES.items():
        if model == k or model.startswith(k + "-"):
            return v
    return None


def call_cost(model: Optional[str], usage: Optional[dict]) -> Optional[float]:
    """호출 1회의 비용(USD)."""
    p = price_for(model)
    if p is None or not isinstance(usage, dict):
        return None
    return sum((usage.get(f) or 0) * p[k] for f, k in _FIELDS.items()) / 1_000_000


def gemini_usage(response: object) -> dict:
    """Gemini 응답의 usage_metadata → Claude와 같은 키. 출력 = 답변 + thinking 토큰 (둘 다 출력 요금)."""
    m = getattr(response, "usage_metadata", None)
    get = (lambda k: getattr(m, k, 0) or 0) if m is not None else (lambda k: 0)
    return {"input_tokens": get("prompt_token_count"),
            "output_tokens": get("candidates_token_count") + get("thoughts_token_count")}


def record_cost(rec: dict) -> Optional[float]:
    """채점 기록 1건(rv2_grader.diagnose 결과)의 비용. 호출별로 실제 응답한 모델 가격을 쓴다.
    호출이 없으면 0. 가격을 모르는 호출이 하나라도 있으면 None."""
    raws = rec.get("raw_responses") or []
    if not raws:
        return 0.0 if not rec.get("n_calls") else call_cost(rec.get("model"), rec.get("usage"))
    total = 0.0
    for raw in raws:
        c = call_cost(raw.get("model") or rec.get("model"), raw.get("usage"))
        if c is None:
            return None
        total += c
    return round(total, 6)


# ─── 서버가 켜진 뒤 누적 (프로세스 메모리, 재시작하면 0) ───
_lock = threading.Lock()
_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))


def add(kind: str, model: Optional[str], usage: Optional[dict], cost: Optional[float], calls: int = 1) -> None:
    """kind: grade / refine 등. 누적 표(/admin/usage)용."""
    key = f"{kind}|{model or '?'}"
    with _lock:
        t = _totals[key]
        t["requests"] += 1
        t["calls"] += calls
        for f in _FIELDS:
            t[f] += (usage or {}).get(f) or 0
        if cost is None:
            t["unpriced_requests"] += 1
        else:
            t["cost_usd"] += cost


def snapshot() -> list:
    with _lock:
        rows = []
        for key, t in sorted(_totals.items()):
            kind, model = key.split("|", 1)
            rows.append({"kind": kind, "model": model, **{k: (round(v, 6) if k == "cost_usd" else int(v))
                                                          for k, v in t.items()}})
        return rows


def reset() -> None:
    with _lock:
        _totals.clear()


def summarize(records: Iterable[dict]) -> dict:
    """채점 기록 여러 건(JSONL 등) → 합계. 실험 비용 정리용."""
    n = calls = 0
    tokens: Dict[str, int] = defaultdict(int)
    cost, unpriced = 0.0, 0
    for r in records:
        n += 1
        calls += r.get("n_calls") or 0
        for f in _FIELDS:
            tokens[f] += (r.get("usage") or {}).get(f) or 0
        c = r.get("cost_usd") if r.get("cost_usd") is not None else record_cost(r)
        if c is None:
            unpriced += 1
        else:
            cost += c
    return {"records": n, "calls": calls, **dict(tokens), "cost_usd": round(cost, 4),
            "cost_per_record_usd": round(cost / (n - unpriced), 5) if n - unpriced else None,
            "unpriced_records": unpriced}
