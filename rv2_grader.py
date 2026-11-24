# coding: utf-8
"""
rv2_grader.py — R v2 진단 (키트 run_eval.py의 서버 이식본)

흐름: 프롬프트 조립(엑셀) → LLM 호출(tool_use, 모델별로 강제 또는 auto) → 기계 검증 → 실패 시 사유 붙여 1회 재시도
      → 결과 + 재현성 필드 + 응답 원문 반환.

README 이식 규칙 대응:
  3. 검증은 코드, 판단은 LLM — validate(): 인용문 원문 대조(환각 차단), 허용 밖 코드 거부, 1회 재시도
  4. 기권 허용 — review_needed / review_reason을 그대로 저장
  5. 재현성 필드 — answer_id, prompt_version, model, timestamp (+ 엑셀 해시, 응답 원문, 토큰)

hongsu-rv2-1.2 (코드북 R-v3 규칙 12 대응):
  - 출력에 response_complete(요구 충족 여부)·missing_criteria(대응 풀이가 없는 기준 번호 리스트) 추가.
    스키마 정의는 prompts/rv2.py, 여기서는 검증·정규화만 한다.
  - validate(): missing_criteria의 번호가 이 문항의 기준 판정표 번호인지 확인.
  - normalize_result(): 모델이 필드를 빼먹어도 저장·채점이 깨지지 않게 형식을 맞춘다.

hongsu-rv2-1.3 (안정화):
  - extract_result(): tool_use 블록이 없으면 텍스트 블록의 JSON을 읽는다 (도구 강제를 거부하는 모델 대응).
  - sanitize(): LLM 응답 정제를 한 곳으로 통합. 고칠 수 있는 형식은 고치고,
    고칠 수 없는 것(문자열 줄, 허용 밖 판정값 등)은 조용히 버리지 않고 issues로 돌려 재시도·검수로 보낸다.
  - 결과 블록이 아예 없어도 예외로 끝내지 않고 1회 재시도한다.
  - rec["extraction"]: 호출마다 결과를 어디서 읽었는지(tool/text/none) 기록 — 스키마 안정성 측정용.

환경변수:
  RV2_MODEL        기본 claude-opus-5-5 (11/24 결정. 이전: claude-haiku-4-5 ← 11/8, claude-opus-4-6)
  RV2_MAX_TOKENS   기본 8000 (실험에서 2000은 thinking 모델이 잘림)
  RV2_TEMPERATURE  지정 안 하면 모델별 기본값(MODEL_DEFAULTS). "none"이면 전송 안 함
  RV2_TOOL_CHOICE  지정 안 하면 모델별 기본값. "tool"=도구 호출 강제, "auto"=강제 안 함
"""
import datetime
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import rv2_codebook
from prompts import rv2 as p_rv2

# 모델별 호출 기본값 (11/11 모델 비교 실험에서 확인한 제약).
#   tool_choice: 도구 호출 강제를 거부하는 모델은 "auto"
#   temperature: temperature를 받지 않는 모델은 None(전송 안 함)
MODEL_DEFAULTS: Dict[str, dict] = {
    "claude-opus-5-5":  {"tool_choice": "auto", "temperature": None},
    "claude-haiku-5-5": {"tool_choice": "tool", "temperature": None},
    "claude-opus-4-6":  {"tool_choice": "tool", "temperature": 0.0},
    "claude-haiku-4-5": {"tool_choice": "tool", "temperature": 0.0},
}
_FALLBACK_DEFAULTS: dict = {"tool_choice": "tool", "temperature": 0.0}


def model_defaults(model: str) -> dict:
    """모델 ID(날짜 접미사 포함)에 맞는 호출 기본값."""
    for k, v in MODEL_DEFAULTS.items():
        if model == k or model.startswith(k + "-"):
            return v
    return _FALLBACK_DEFAULTS


RV2_MODEL = os.environ.get("RV2_MODEL", "claude-opus-5-5")
RV2_MAX_TOKENS = int(os.environ.get("RV2_MAX_TOKENS", "8000"))
_defaults = model_defaults(RV2_MODEL)

_t = os.environ.get("RV2_TEMPERATURE")
if _t is None:
    RV2_TEMPERATURE = _defaults["temperature"]
else:
    RV2_TEMPERATURE = None if _t.strip().lower() == "none" else float(_t)

RV2_TOOL_CHOICE = (os.environ.get("RV2_TOOL_CHOICE") or _defaults["tool_choice"]).strip().lower()
if RV2_TOOL_CHOICE not in ("tool", "auto"):
    RV2_TOOL_CHOICE = _defaults["tool_choice"]

# 채점 동작(정제·검증·재시도 로직) 버전. 프롬프트가 같아도 이 값이 다르면 결과를 직접 비교하지 않는다.
GRADER_VERSION = "rv2-grader-1.3"

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def norm(s: str) -> str:
    """키트 run_eval.norm()과 동일: 공백 전부 제거."""
    return re.sub(r"\s+", "", s or "")


def _to_int(v: Any) -> Optional[int]:
    """'5', 5, '⑤', '기준5' → 5. 해석 못 하면 None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):  # 3.0 → 3, 3.5처럼 정수가 아니면 해석 불가
        return int(v) if v.is_integer() else None
    s = str(v or "").strip()
    if re.fullmatch(r"\d+\.\d+", s):
        f = float(s)
        return int(f) if f.is_integer() else None
    if s and s[0] in _CIRCLED:
        return _CIRCLED[s[0]]
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


_CIRCLED = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩")}


def normalize_result(parsed: dict) -> dict:
    """response_complete / missing_criteria 형식 정리.
    - missing_criteria: 정수 리스트(중복 제거·정렬). 없으면 [].
    - response_complete: bool. 모델이 빼먹으면 None(판단 없음)으로 둔다 — True로 추정하지 않는다."""
    if not isinstance(parsed, dict):
        return parsed
    mc = parsed.get("missing_criteria")
    if not isinstance(mc, list):
        mc = [] if mc in (None, "", "null") else [mc]
    parsed["missing_criteria"] = sorted({n for n in (_to_int(x) for x in mc) if n is not None})
    rc = parsed.get("response_complete")
    if isinstance(rc, str):
        rc = {"true": True, "false": False}.get(rc.strip().lower())
    parsed["response_complete"] = rc if isinstance(rc, bool) else None
    return parsed


# ────────────────────────────────
# 응답 정제 (hongsu-rv2-1.3)
# ────────────────────────────────

# 코드 필드(code, main_error_code)에서 '값 없음'으로 보는 문자열
_NULL_WORDS = {"", "null", "none", "nil", "n/a", "na", "-", "없음", "해당없음", "해당 없음"}
# 텍스트 필드(quote, reason)는 학생이 실제로 '없음', '-'라고 쓴 인용일 수 있어 최소한만 null로 본다
_TEXT_NULL_WORDS = {"", "null", "none"}
_JUDGMENTS = {"ok", "wrong", "depends"}
_J_LEVELS = {"J0", "J1", "J2"}
_INPUT_STATUS = {"confirmed", "not_attempted", "unreadable"}


def _to_bool(v: Any) -> Optional[bool]:
    """True/False/'true'/'false'/1/0 → bool, 해석 못 하면 None."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        return {"true": True, "false": False, "yes": True, "no": False,
                "1": True, "0": False}.get(v.strip().lower())
    return None


def _to_null_or_str(v: Any, null_words: set = _NULL_WORDS) -> Optional[str]:
    """None·'null'·'' 같은 빈 값 → None, 그 외 → 문자열."""
    if v is None:
        return None
    s = str(v)
    return None if s.strip().lower() in null_words else s


def _maybe_json(v: Any) -> Any:
    """문자열로 온 JSON 객체·배열을 푼다. 못 풀면 원래 값."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                pass
    return v


def _clean_line(ln: dict, pos: int, issues: list) -> dict:
    """줄 하나의 필드 형식을 맞춘다. pos = lines 안 위치(1부터, 사유 표시용)."""
    out = dict(ln)
    no = _to_int(ln.get("no"))
    if no is None:
        issues.append(f"lines[{pos}]: 줄 번호(no) 없음 또는 해석 불가: {ln.get('no')!r}")
    out["no"] = no
    j = str(ln.get("judgment") or "").strip().lower()
    if j not in _JUDGMENTS:
        issues.append(f"line {no if no is not None else f'#{pos}'}: judgment 값이 ok/wrong/depends가 아님: "
                      f"{ln.get('judgment')!r}")
    out["judgment"] = j or None
    out["student_text"] = "" if ln.get("student_text") is None else str(ln.get("student_text"))
    for k in ("quote", "reason"):
        if k in ln:
            out[k] = _to_null_or_str(ln.get(k), _TEXT_NULL_WORDS)
    if "code" in ln:
        c = _to_null_or_str(ln.get("code"))
        out["code"] = c.strip().upper() if c else None
    if "matched_criterion" in ln:
        out["matched_criterion"] = _to_int(ln.get("matched_criterion"))
    if "depends_on" in ln:
        dep = _maybe_json(ln.get("depends_on"))
        if isinstance(dep, list):  # 원인 줄이 여러 개면 리스트 그대로 둔다 (정보를 버리지 않음)
            out["depends_on"] = [n for n in (_to_int(x) for x in dep) if n is not None] or None
        else:
            out["depends_on"] = _to_int(dep) if dep is not None else None
    return out


def sanitize(parsed: Any, mode: str = "with_codebook") -> Tuple[Optional[dict], list]:
    """LLM 응답 정제 (단일 진입점).

    반환: (정제된 dict, issues 리스트)
      - 고칠 수 있는 형식(문자열 불리언, 'null' 문자열, 대소문자, '③' 같은 번호)은 고친다.
      - 고칠 수 없는 것은 버리지 않고 issues에 사유를 남긴다 → diagnose()가 1회 재시도, 그래도 남으면 검수.
    """
    issues = []
    parsed = _maybe_json(parsed)
    if not isinstance(parsed, dict):
        return {}, [f"응답이 객체(dict)가 아님: {type(parsed).__name__}"]
    out = dict(parsed)

    # lines
    lines = _maybe_json(parsed.get("lines"))
    if lines is None:
        issues.append("lines 없음")
        lines = []
    elif not isinstance(lines, list):
        issues.append(f"lines가 리스트가 아님: {type(lines).__name__}")
        lines = []
    clean_lines = []
    for i, ln in enumerate(lines, 1):
        ln = _maybe_json(ln)
        if not isinstance(ln, dict):
            preview = str(ln)[:40]
            issues.append(f"lines[{i}]: 줄이 객체가 아님 ({type(ln).__name__}: {preview!r})")
            continue
        clean_lines.append(_clean_line(ln, i, issues))
    out["lines"] = clean_lines

    # 공통 필드
    mrc = _to_bool(parsed.get("math_result_correct"))
    if mrc is None:
        issues.append(f"math_result_correct 해석 불가: {parsed.get('math_result_correct')!r}")
    out["math_result_correct"] = mrc

    st = str(parsed.get("input_status") or "confirmed").strip().lower()
    if st not in _INPUT_STATUS:
        issues.append(f"input_status 값 오류: {parsed.get('input_status')!r}")
        st = "confirmed"
    out["input_status"] = st

    if mode == "with_codebook":
        mec = _to_null_or_str(parsed.get("main_error_code"))
        out["main_error_code"] = mec.strip().upper() if mec else None

        jl = _to_null_or_str(parsed.get("justification_level"))
        jl = jl.strip() if jl else jl
        if jl is None:
            issues.append("justification_level 없음")
        elif jl.upper() in _J_LEVELS:
            jl = jl.upper()
        elif jl.lower().replace(" ", "_") == "not_applicable":
            jl = "not_applicable"
        else:
            issues.append(f"justification_level 값 오류: {jl!r}")
        out["justification_level"] = jl

        rn = _to_bool(parsed.get("review_needed"))
        out["review_needed"] = bool(rn) if rn is not None else False
        rr = parsed.get("review_reason")
        out["review_reason"] = "" if rr is None else str(rr)

    out = normalize_result(out)  # response_complete / missing_criteria
    return out, issues


def extract_result(content_blocks: list) -> Tuple[Optional[dict], str]:
    """응답 블록에서 결과 dict와 출처를 꺼낸다.
    tool_use 블록 우선 → 없으면 텍스트 블록의 JSON(```json 펜스 포함) → 둘 다 없으면 (None, 'none')."""
    texts = []
    for b in content_blocks or []:
        btype = getattr(b, "type", None) if not isinstance(b, dict) else b.get("type")
        if btype == "tool_use":
            inp = getattr(b, "input", None) if not isinstance(b, dict) else b.get("input")
            return (dict(inp) if isinstance(inp, dict) else inp), "tool"
        if btype == "text":
            texts.append(getattr(b, "text", None) if not isinstance(b, dict) else b.get("text"))
    for t in texts:
        if not t:
            continue
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
        cand = m.group(1) if m else None
        if cand is None and "{" in t and "}" in t:
            cand = t[t.index("{"): t.rindex("}") + 1]
        if cand:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj, "text"
            except (json.JSONDecodeError, ValueError):
                continue
    return None, "none"


def validate(parsed: dict, answer_text: str, allowed: list, with_codes: bool,
             criteria_nos: Optional[list] = None) -> list:
    """키트 run_eval.validate()와 동일한 기계 검증 + missing_criteria 번호 검사.
    실패 사유 목록 (빈 리스트면 통과)."""
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
    if criteria_nos:
        bad = [n for n in (parsed.get("missing_criteria") or []) if n not in criteria_nos]
        if bad:
            issues.append(f"missing_criteria에 기준 판정표에 없는 번호: {bad} (가능: {criteria_nos})")
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


def _dump_content(resp: Any) -> list:
    """응답 원문 보존 (블록 단위). SDK 객체 → dict."""
    out = []
    for b in getattr(resp, "content", []) or []:
        try:
            out.append(b.model_dump())
        except Exception:
            out.append({"type": getattr(b, "type", "?"), "repr": repr(b)})
    return out


def _call(client: Any, model: str, system: str, user: str, tool: dict, max_tokens: int,
          temperature: Optional[float], tool_choice: Optional[str] = None) -> Tuple[Optional[dict], dict]:
    tool_choice = tool_choice or RV2_TOOL_CHOICE
    kw = dict(model=model, max_tokens=max_tokens, system=system,
              tools=[tool], messages=[{"role": "user", "content": user}])
    if tool_choice == "tool":
        kw["tool_choice"] = {"type": "tool", "name": tool["name"]}
    else:
        kw["tool_choice"] = {"type": "auto"}
    if temperature is not None:
        kw["temperature"] = temperature
    resp = client.messages.create(**kw)
    u = getattr(resp, "usage", None)
    usage = {k: (getattr(u, k, 0) or 0) for k in USAGE_KEYS}
    parsed, source = extract_result(getattr(resp, "content", []))
    raw = {"stop_reason": resp.stop_reason, "usage": usage, "content": _dump_content(resp),
           "extraction": source, "model": getattr(resp, "model", None), "ts": _now()}
    return parsed, raw


def diagnose(client: Any, problem_id: str, answer_text: str, mode: str = "with_codebook",
             model: Optional[str] = None, max_tokens: Optional[int] = None,
             temperature: Any = "default", tool_choice: Optional[str] = None) -> dict:
    """R v2 진단 1건. 예외를 던지지 않고 항상 결과 dict를 돌려준다 (실패도 원문과 함께 기록)."""
    assert mode in ("grade_only", "with_codebook")
    model = model or RV2_MODEL
    max_tokens = max_tokens or RV2_MAX_TOKENS
    # 서버 기본 모델이면 환경변수 반영값, 다른 모델을 직접 지정했으면 그 모델의 기본값
    md = ({"tool_choice": RV2_TOOL_CHOICE, "temperature": RV2_TEMPERATURE} if model == RV2_MODEL
          else model_defaults(model))
    temperature = md["temperature"] if temperature == "default" else temperature
    tool_choice = tool_choice or md["tool_choice"]
    with_codes = mode == "with_codebook"
    mats = rv2_codebook.get()

    built = p_rv2.build_prompt(mats, problem_id, mode)
    tool = p_rv2.tool_schema(mode)
    user = built["user_template"].replace("{STUDENT_ANSWER}", answer_text)
    criteria_nos = sorted({n for n in (_to_int(r.get("line_no")) for r in rv2_codebook.lines_for(problem_id))
                           if n is not None})

    rec = {
        "problem_id": problem_id, "mode": mode, "model": model,
        "prompt_version": built["version"], "grader_version": GRADER_VERSION,
        "codebook_version": mats.get("codebook_version"),
        "xlsx_sha8": mats.get("xlsx_sha8"), "allowed_codes": built["allowed_codes"],
        "answer_text": answer_text, "max_tokens": max_tokens, "temperature": temperature,
        "ts": _now(), "raw_responses": [], "usage": {k: 0 for k in USAGE_KEYS},
        "n_calls": 0, "stop_reasons": [], "result": None, "validation_issues": [], "error": None,
        "tool_choice": tool_choice,
        "response_models": [],     # API가 실제로 응답한 모델 ID (요청한 model과 일치하는지 확인용)
        "extraction": [],          # 호출마다 tool / text / none
        "first_issues": [],        # 첫 호출의 정제·검증 사유 (재시도로 해결됐어도 기록 — 안정성 측정용)
    }

    def _one(u: str) -> Tuple[Optional[dict], Optional[str]]:
        parsed, raw = _call(client, model, built["system"], u, tool, max_tokens, temperature,
                            tool_choice=tool_choice)
        rec["raw_responses"].append(raw)
        rec["n_calls"] += 1
        rec["stop_reasons"].append(raw["stop_reason"])
        rec["extraction"].append(raw["extraction"])
        rec["response_models"].append(raw.get("model"))
        for k in USAGE_KEYS:
            rec["usage"][k] += raw["usage"].get(k, 0)
        return parsed, raw["stop_reason"]

    def _check(parsed: Optional[dict], stop_reason: Optional[str]) -> Tuple[Optional[dict], list]:
        """정제 + 검증. 결과 블록이 없으면 (None, [사유])."""
        if parsed is None:
            return None, [f"결과 블록 없음 (stop_reason={stop_reason}) — 지정된 도구 형식(JSON)으로 출력할 것"]
        clean, s_issues = sanitize(parsed, mode)
        if not clean:
            return None, s_issues
        v_issues = validate(clean, answer_text, built["allowed_codes"], with_codes, criteria_nos)
        return clean, s_issues + v_issues

    try:
        clean, issues = _check(*_one(user))
        rec["first_issues"] = list(issues)
        if issues:  # 키트와 동일: 실패 사유를 붙여 1회 재시도 (결과 블록 없음·정제 실패 포함)
            retry_user = user + "\n\n[이전 출력의 문제점 — 수정하여 다시 출력]\n- " + "\n- ".join(issues)
            clean2, issues2 = _check(*_one(retry_user))
            if clean2 is not None:
                clean, issues = clean2, issues2
            elif clean is None:
                issues = issues2
        if clean is None:
            rec["error"] = "결과 블록 없음 (재시도 후에도): " + "; ".join(issues)
        rec["result"] = clean
        rec["validation_issues"] = issues
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def classify_outcome(rec: dict) -> Tuple[str, list]:
    """진단 기록 → (outcome, review_sources).

    outcome (우선순위 순):
      error          결과 없음 (재시도 후에도 결과 블록 없음, API 오류)
      not_attempted  백지
      unreadable     판독 불가
      review         검수 필요 — 모델 기권, 재시도 후에도 남은 형식·검증 문제, 출력 잘림
      incomplete     미완 답안 (response_complete=false, R-v3 규칙 12)
      graded         채점 완료
    review_sources: review로 간 이유 목록 (model / validation / max_tokens). 기권 분류·현장 처리에 쓴다.
    """
    res = rec.get("result") or {}
    if rec.get("error") or not res:
        return "error", ["error"]
    st = res.get("input_status")
    if st in ("not_attempted", "unreadable"):
        return st, []
    sources = []
    if res.get("review_needed"):
        sources.append("model")
    if rec.get("validation_issues"):
        sources.append("validation")
    if "max_tokens" in (rec.get("stop_reasons") or []):
        sources.append("max_tokens")
    if sources:
        return "review", sources
    if res.get("response_complete") is False:
        return "incomplete", []
    return "graded", []


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
