# coding: utf-8
"""
rv2_grader.py — R v2 진단 (키트 run_eval.py의 서버 이식본)

흐름: 프롬프트 조립(엑셀) → LLM 호출(tool_use 강제) → 기계 검증 → 실패 시 사유 붙여 1회 재시도
      → 결과 + 재현성 필드 + 응답 원문 반환.

README 이식 규칙 대응:
  3. 검증은 코드, 판단은 LLM — validate(): 인용문 원문 대조(환각 차단), 허용 밖 코드 거부, 1회 재시도
  4. 기권 허용 — review_needed / review_reason을 그대로 저장
  5. 재현성 필드 — answer_id, prompt_version, model, timestamp (+ 엑셀 해시, 응답 원문, 토큰)

환경변수:
  RV2_MODEL        기본 claude-opus-4-6 (학생 서버 모델, 실험 조건과 동일)
  RV2_MAX_TOKENS   기본 8000 (실험에서 2000은 thinking 모델이 잘림)
  RV2_TEMPERATURE  기본 0.0, "none"이면 전송 안 함 (Fable 5처럼 temperature를 거부하는 모델용)
"""
import datetime
import os
import re

import rv2_codebook
from prompts import rv2 as p_rv2

RV2_MODEL = os.environ.get("RV2_MODEL", "claude-opus-4-6")
RV2_MAX_TOKENS = int(os.environ.get("RV2_MAX_TOKENS", "8000"))
_t = os.environ.get("RV2_TEMPERATURE", "0.0")
RV2_TEMPERATURE = None if _t.lower() == "none" else float(_t)

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def norm(s: str) -> str:
    """키트 run_eval.norm()과 동일: 공백 전부 제거."""
    return re.sub(r"\s+", "", s or "")


def validate(parsed: dict, answer_text: str, allowed: list, with_codes: bool) -> list:
    """키트 run_eval.validate()와 동일한 기계 검증. 실패 사유 목록 (빈 리스트면 통과)."""
    issues = []
    na = norm(answer_text)
    for ln in parsed.get("lines", []) or []:
        if not isinstance(ln, dict):
            continue
        if ln.get("judgment") == "wrong":
            q = ln.get("quote") or ""
            if not q:
                issues.append(f"line {ln.get('no')}: wrong인데 quote 없음")
            elif norm(q) not in na:
                issues.append(f"line {ln.get('no')}: quote가 답안 원문에 없음")
            if with_codes:
                c = ln.get("code")
                if c and c not in allowed:
                    issues.append(f"line {ln.get('no')}: 허용 밖 코드 {c}")
    if with_codes:
        mec = parsed.get("main_error_code")
        if mec and mec not in allowed:
            issues.append(f"main_error_code 허용 밖: {mec}")
    return issues


def steps_to_answer_text(steps: list) -> str:
    """OCR 확인된 단계 [{index, text}] → '1) ...' 줄 단위 답안. 줄 번호 = 단계 번호가 되도록."""
    out = []
    for i, s in enumerate(steps or []):
        idx = s.get("index", i + 1)
        txt = (s.get("text") or "").strip()
        if txt:
            out.append(f"{idx}) {txt}")
    return "\n".join(out)


def _dump_content(resp) -> list:
    """응답 원문 보존 (블록 단위). SDK 객체 → dict."""
    out = []
    for b in getattr(resp, "content", []) or []:
        try:
            out.append(b.model_dump())
        except Exception:
            out.append({"type": getattr(b, "type", "?"), "repr": repr(b)})
    return out


def _call(client, model, system, user, tool, max_tokens, temperature):
    kw = dict(model=model, max_tokens=max_tokens, system=system,
              tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
              messages=[{"role": "user", "content": user}])
    if temperature is not None:
        kw["temperature"] = temperature
    resp = client.messages.create(**kw)
    u = getattr(resp, "usage", None)
    usage = {k: (getattr(u, k, 0) or 0) for k in USAGE_KEYS}
    parsed = None
    for b in resp.content:
        if getattr(b, "type", None) == "tool_use":
            parsed = dict(b.input)
            break
    raw = {"stop_reason": resp.stop_reason, "usage": usage, "content": _dump_content(resp), "ts": _now()}
    return parsed, raw


def diagnose(client, problem_id: str, answer_text: str, mode: str = "with_codebook",
             model: str = None, max_tokens: int = None, temperature="default") -> dict:
    """R v2 진단 1건. 예외를 던지지 않고 항상 결과 dict를 돌려준다 (실패도 원문과 함께 기록)."""
    assert mode in ("grade_only", "with_codebook")
    model = model or RV2_MODEL
    max_tokens = max_tokens or RV2_MAX_TOKENS
    temperature = RV2_TEMPERATURE if temperature == "default" else temperature
    with_codes = mode == "with_codebook"
    mats = rv2_codebook.get()

    built = p_rv2.build_prompt(mats, problem_id, mode)
    tool = p_rv2.tool_schema(mode)
    user = built["user_template"].replace("{STUDENT_ANSWER}", answer_text)

    rec = {
        "problem_id": problem_id, "mode": mode, "model": model,
        "prompt_version": built["version"], "codebook_version": mats.get("codebook_version"),
        "xlsx_sha8": mats.get("xlsx_sha8"), "allowed_codes": built["allowed_codes"],
        "answer_text": answer_text, "max_tokens": max_tokens, "temperature": temperature,
        "ts": _now(), "raw_responses": [], "usage": {k: 0 for k in USAGE_KEYS},
        "n_calls": 0, "stop_reasons": [], "result": None, "validation_issues": [], "error": None,
    }

    def _one(u):
        parsed, raw = _call(client, model, built["system"], u, tool, max_tokens, temperature)
        rec["raw_responses"].append(raw)
        rec["n_calls"] += 1
        rec["stop_reasons"].append(raw["stop_reason"])
        for k in USAGE_KEYS:
            rec["usage"][k] += raw["usage"].get(k, 0)
        if parsed is None:
            raise ValueError(f"tool_use 블록 없음 (stop_reason={raw['stop_reason']})")
        return parsed

    try:
        parsed = _one(user)
        issues = validate(parsed, answer_text, built["allowed_codes"], with_codes)
        if issues:  # 키트와 동일: 실패 사유를 붙여 1회 재시도
            retry_user = user + "\n\n[이전 출력의 문제점 — 수정하여 다시 출력]\n- " + "\n- ".join(issues)
            parsed = _one(retry_user)
            issues = validate(parsed, answer_text, built["allowed_codes"], with_codes)
        rec["result"] = parsed
        rec["validation_issues"] = issues
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def to_server_steps(result: dict) -> list:
    """기존 프론트 호환용: lines → steps(index/text/status/explanation/depends_on_line_id)."""
    steps = []
    for ln in (result or {}).get("lines", []) or []:
        if not isinstance(ln, dict):
            continue
        dep = ln.get("depends_on")
        if isinstance(dep, list):
            dep = dep[0] if dep else None
        steps.append({
            "index": ln.get("no"),
            "text": ln.get("student_text", ""),
            "status": ln.get("judgment"),
            "explanation": ln.get("reason") or "",
            "depends_on_line_id": dep,
            "code": ln.get("code"),
            "quote": ln.get("quote"),
        })
    return steps
