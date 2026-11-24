# coding: utf-8
"""입력 검증과 요청 횟수 제한.

입력 검증: 잘못된 요청은 500(서버 오류)이 아니라 400(요청 오류)과 이유로 돌려준다.
  - BadInput을 던지면 app.py의 처리기가 {"success": false, "error": 이유}, 400으로 바꾼다.

요청 횟수 제한 (@rate_limited): 비용이 드는 엔드포인트(/grade, /ocr, /refine, /measure_consistency)에만.
  - 학생별(student_id) 1분당 RATE_LIMIT_PER_MIN회, IP별 1분당 RATE_LIMIT_IP_PER_MIN회. 엔드포인트마다 따로 센다.
  - 교실은 학생 여럿이 IP 하나(학교 공유기)를 쓰므로 IP 한도는 넉넉하게 둔다.
  - 0이면 끈다. 서버 안에서 다시 부르는 요청(/measure_consistency → /grade)은 세지 않는다.
  - gunicorn 워커 1개 기준(Procfile)의 메모리 카운터다. 워커를 늘리면 워커마다 따로 센다.
"""
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable, Deque, Dict, Optional

from flask import jsonify, request

from hongsu.config import RATE_LIMIT_IP_PER_MIN, RATE_LIMIT_PER_MIN

INTERNAL_FLAG = "hongsu.internal"   # 서버 내부 호출 표시 (environ_base로만 넣을 수 있어 밖에서 꾸밀 수 없다)


class BadInput(ValueError):
    """요청 형식 오류 → 400."""


def json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadInput("JSON 객체 본문이 필요함 (Content-Type: application/json)")
    return data


def get_str(data: dict, key: str, default: str = "", max_len: Optional[int] = None) -> str:
    v = data.get(key, default)
    if v is None:
        v = default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = str(v)
    if not isinstance(v, str):
        raise BadInput(f"{key}: 문자열이어야 함")
    if max_len is not None and len(v) > max_len:
        raise BadInput(f"{key}: 너무 김 ({len(v)}자, 최대 {max_len}자)")
    return v


def get_int(data: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = data.get(key, default)
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise BadInput(f"{key}: 정수여야 함 ({v!r})")
    if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
        raise BadInput(f"{key}: 정수여야 함 ({v!r})")
    if not lo <= n <= hi:
        raise BadInput(f"{key}: {lo}~{hi} 사이여야 함 ({n})")
    return n


# ─── 요청 횟수 제한 ───
_hits: Dict[str, Deque[float]] = defaultdict(deque)
_lock = threading.Lock()
WINDOW_S = 60.0


def reset() -> None:
    """테스트용: 카운터 비우기."""
    with _lock:
        _hits.clear()


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else "") or request.remote_addr or "?"


def rate_limited(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if request.environ.get(INTERNAL_FLAG):
            return fn(*args, **kwargs)
        body = request.get_json(silent=True)
        sid = body.get("student_id") if isinstance(body, dict) else None
        checks = []
        if RATE_LIMIT_PER_MIN > 0 and isinstance(sid, (str, int)) and str(sid) not in ("", "anonymous"):
            checks.append((f"{request.path}|s:{sid}", RATE_LIMIT_PER_MIN, "학생"))
        if RATE_LIMIT_IP_PER_MIN > 0:
            checks.append((f"{request.path}|ip:{_client_ip()}", RATE_LIMIT_IP_PER_MIN, "IP"))
        now = time.monotonic()
        with _lock:
            for key, limit, who in checks:   # 하나라도 넘으면 아무것도 기록하지 않고 거절
                q = _hits[key]
                while q and now - q[0] >= WINDOW_S:
                    q.popleft()
                if len(q) >= limit:
                    wait = int(WINDOW_S - (now - q[0])) + 1
                    resp = jsonify({"success": False, "error": f"요청이 너무 많음 ({who}별 1분 {limit}회). "
                                                              f"{wait}초 뒤 다시 시도"})
                    resp.status_code = 429
                    resp.headers["Retry-After"] = str(wait)
                    return resp
            for key, _limit, _who in checks:
                _hits[key].append(now)
        return fn(*args, **kwargs)
    return wrapper
