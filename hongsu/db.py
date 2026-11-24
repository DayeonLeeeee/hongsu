# coding: utf-8
"""PostgreSQL 저장 (Railway Postgres 등). 환경변수 DATABASE_URL이 있으면 Supabase 대신 여기에 저장한다.

- 테이블이 없으면 처음 쓸 때 만든다 (CREATE TABLE IF NOT EXISTS). 컬럼이 늘면 ADD COLUMN IF NOT EXISTS로 맞춘다.
  → Railway에서 SQL을 따로 실행할 필요가 없다.
- 저장할 때마다 새로 연결한다 (요청 수가 적고, DB 재시작·연결 끊김에 강하다).
"""
import threading
from typing import Any

from hongsu.config import DATABASE_URL
from hongsu.logging_setup import log

# rv2_gradings 컬럼 → 타입. 순서는 sql/2026-11-24_rv2_gradings_create.sql과 같다.
RV2_COLUMNS = {
    "answer_id": "text", "problem_id": "text", "student_id": "text", "attempt_count": "integer",
    "mode": "text", "input_status": "text", "answer_text": "text",
    "lines": "jsonb", "main_error_code": "text", "justification_level": "text",
    "math_result_correct": "boolean", "review_needed": "boolean", "review_reason": "text",
    "response_complete": "boolean", "missing_criteria": "jsonb", "validation_issues": "jsonb", "error": "text",
    "prompt_version": "text", "codebook_version": "text", "xlsx_sha8": "text", "allowed_codes": "jsonb",
    "model": "text", "n_calls": "integer", "usage": "jsonb", "stop_reasons": "jsonb", "raw_responses": "jsonb",
    "request_id": "text", "outcome": "text", "review_sources": "jsonb", "grader_version": "text",
    "tool_choice": "text", "response_models": "jsonb", "extraction": "jsonb", "first_issues": "jsonb",
    "reveal_answer": "boolean", "feedback": "jsonb", "git_commit": "text", "cost_usd": "numeric",
}

_DDL = ["CREATE TABLE IF NOT EXISTS rv2_gradings ("
        "id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
        "created_at timestamptz NOT NULL DEFAULT now(), "
        + ", ".join(f"{c} {t}" for c, t in RV2_COLUMNS.items()) + ")"]
_DDL += [f"ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS {c} {t}" for c, t in RV2_COLUMNS.items()]
_DDL += ["CREATE INDEX IF NOT EXISTS rv2_gradings_problem_idx ON rv2_gradings (problem_id)",
         "CREATE INDEX IF NOT EXISTS rv2_gradings_student_idx ON rv2_gradings (student_id)",
         "CREATE INDEX IF NOT EXISTS rv2_gradings_created_idx ON rv2_gradings (created_at)"]

_ready = False
_lock = threading.Lock()


def enabled() -> bool:
    return bool(DATABASE_URL)


def _connect() -> Any:
    import psycopg
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def ensure_table() -> None:
    """테이블·컬럼·인덱스를 맞춘다 (프로세스당 한 번)."""
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with _connect() as conn:
            for stmt in _DDL:
                conn.execute(stmt)
        _ready = True
        log.info("[DB] rv2_gradings 테이블 확인 완료")


def _value(col: str, v: Any) -> Any:
    if v is None:
        return None
    if RV2_COLUMNS[col] == "jsonb":
        from psycopg.types.json import Jsonb
        return Jsonb(v)
    if RV2_COLUMNS[col] == "text" and not isinstance(v, str):
        return str(v)
    return v


def insert_rv2(row: dict) -> None:
    """rv2_gradings에 1행. 모르는 키는 버린다. 실패하면 예외를 그대로 올린다."""
    ensure_table()
    cols = [c for c in RV2_COLUMNS if c in row]
    sql = (f"INSERT INTO rv2_gradings ({', '.join(cols)}) "
           f"VALUES ({', '.join(['%s'] * len(cols))})")
    with _connect() as conn:
        conn.execute(sql, [_value(c, row[c]) for c in cols])


def usage_by_day(days: int = 14) -> list:
    """/admin/usage용: 날짜(서울)·모델별 채점 수, 호출 수, 토큰, 비용."""
    ensure_table()
    sql = """
        SELECT to_char(created_at AT TIME ZONE 'Asia/Seoul', 'YYYY-MM-DD') AS day, model,
               count(*), coalesce(sum(n_calls), 0),
               coalesce(sum((usage->>'input_tokens')::bigint), 0),
               coalesce(sum((usage->>'output_tokens')::bigint), 0),
               coalesce(sum(cost_usd), 0), count(*) FILTER (WHERE cost_usd IS NULL)
        FROM rv2_gradings
        WHERE created_at >= now() - make_interval(days => %s)
        GROUP BY 1, 2 ORDER BY 1 DESC, 2"""
    with _connect() as conn:
        rows = conn.execute(sql, [days]).fetchall()
    return [{"day": d, "model": m, "gradings": n, "calls": int(c), "input_tokens": int(i), "output_tokens": int(o),
             "cost_usd": round(float(cost), 6), "unpriced": int(up)} for d, m, n, c, i, o, cost, up in rows]


def check() -> dict:
    """/health?deep=1용."""
    try:
        ensure_table()
        with _connect() as conn:
            n = conn.execute("SELECT count(*) FROM rv2_gradings").fetchone()[0]
        return {"ok": True, "backend": "postgres", "rows": n}
    except Exception as e:
        return {"ok": False, "backend": "postgres", "error": f"{type(e).__name__}: {str(e)[:200]}"}
