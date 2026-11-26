# coding: utf-8
"""로깅과 요청별 request_id.

모든 로그 줄에 request_id를 붙인다. 요청 밖(서버 시작 등)은 "-".
ContextVar라서 OCR 병렬 호출 스레드에도 같은 ID가 이어진다 (ocr.submit_with_context에서 컨텍스트 복사).
"""
import contextvars
import logging
import os
import re
import uuid
from typing import Optional

from flask import Flask, Response, request

_request_id = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("hongsu")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(message)s"))
        h.addFilter(_RequestIdFilter())
        logger.addHandler(h)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


log = _setup_logging()


# request_id는 요청별 environ에 둔다. flask.g는 앱 컨텍스트 단위라, 요청 안에서 내부 요청을 보내면
# (/measure_consistency → /grade) 안쪽 요청이 바깥 값을 덮어써 끝날 때 에러가 났다.
_RID_KEY = "hongsu.request_id"
_RID_TOKEN_KEY = "hongsu.request_id_token"


def current_request_id() -> Optional[str]:
    """지금 처리 중인 요청의 ID (요청 밖이면 None)."""
    rid = _request_id.get()
    return None if rid == "-" else rid


def _assign_request_id() -> None:
    rid = re.sub(r"[^A-Za-z0-9_.:-]", "", request.headers.get("X-Request-ID") or "")[:64] or uuid.uuid4().hex[:12]
    request.environ[_RID_KEY] = rid
    request.environ[_RID_TOKEN_KEY] = _request_id.set(rid)


def _return_request_id(resp: Response) -> Response:
    rid = request.environ.get(_RID_KEY)
    if rid:
        resp.headers["X-Request-ID"] = rid
    return resp


def _clear_request_id(exc: Optional[BaseException]) -> None:
    tok = request.environ.pop(_RID_TOKEN_KEY, None)
    if tok is not None:
        _request_id.reset(tok)


def init_app(app: Flask) -> None:
    """요청 시작·응답·종료 훅 등록."""
    app.before_request(_assign_request_id)
    app.after_request(_return_request_id)
    app.teardown_request(_clear_request_id)
