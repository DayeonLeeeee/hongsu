# coding: utf-8
"""저장소: 문항·결과 인메모리 DB, rv2_gradings 저장, 엑셀 문항 로드.

인메모리 DB는 서버 재시작 시 초기화된다. 채점 결과의 영구 저장은 rv2_gradings 테이블:
DATABASE_URL이 있으면 PostgreSQL(Railway, hongsu/db.py), 없으면 Supabase.
"""
import json
import threading
from collections import deque
from typing import Dict

import rv2_codebook
from hongsu import clients, db
from hongsu.logging_setup import log
from hongsu.utils import now_iso

problems_db: Dict[str, dict] = {}   # id -> problem dict
results_db: Dict[str, dict] = {}    # id -> 학생 결과 dict (인메모리, /results)
_next_id = 1


def gen_id() -> str:
    global _next_id
    pid = f"P{_next_id:04d}"
    _next_id += 1
    return pid


# ─── Supabase 저장: R v2 진단 (rv2_gradings 테이블) ───
RV2_SAVE_KEYS = ["answer_id", "problem_id", "student_id", "attempt_count", "mode", "input_status",
                 "answer_text", "lines", "main_error_code", "justification_level", "math_result_correct",
                 "review_needed", "review_reason", "response_complete", "missing_criteria",
                 "validation_issues", "error", "prompt_version", "codebook_version", "xlsx_sha8",
                 "allowed_codes", "model", "n_calls", "usage", "stop_reasons", "raw_responses"]
RV2_SAVE_KEYS_V13 = ["request_id", "outcome", "review_sources", "grader_version", "tool_choice",
                     "response_models", "extraction", "first_issues",
                     "reveal_answer", "feedback",   # 학생이 실제로 본 것 (공개 기준·피드백이 바뀌어도 복원 가능)
                     "git_commit", "cost_usd",
                     "code_candidates"]   # 프롬프트 1.8~ (sql/2026-11-26_rv2_gradings_code_candidates.sql)

# DB에 못 쓴 행: 메모리에 잡아 두고 다음 저장(또는 /health?deep=1) 때 다시 넣는다.
# 서버가 재시작되면 사라지므로, 행 전체를 로그에도 남긴다 ("[DB 보류 행]" 줄로 복구 가능).
_pending: "deque[dict]" = deque(maxlen=5000)
_pending_lock = threading.Lock()


def pending_count() -> int:
    return len(_pending)


def flush_pending() -> int:
    """보류 행을 순서대로 다시 넣는다. 넣은 수를 돌려준다. 하나라도 실패하면 거기서 멈춘다."""
    done = 0
    with _pending_lock:
        while _pending:
            try:
                db.insert_rv2(_pending[0])
            except Exception as e:
                log.warning(f"[DB] 보류 행 다시 넣기 실패 (남은 {len(_pending)}건): {type(e).__name__}: {e}")
                break
            _pending.popleft()
            done += 1
    if done:
        log.info(f"[DB] 보류 행 {done}건 저장 완료 (남은 {len(_pending)}건)")
    return done


def save_rv2(r: dict) -> None:
    """R v2 진단 1건 저장. 응답 원문(raw_responses)까지 저장. 실패해도 서버는 계속 동작."""
    if db.enabled():
        row = {k: r.get(k) for k in RV2_SAVE_KEYS + RV2_SAVE_KEYS_V13}
        if _pending:
            flush_pending()
        try:
            if _pending:   # 앞선 행이 아직 못 들어갔으면 순서를 지키려고 뒤에 붙인다
                raise RuntimeError("앞선 보류 행이 남아 있음")
            db.insert_rv2(row)
            log.info(f"[DB] rv2_gradings 저장 완료 ({r.get('answer_id')})")
        except Exception as e:
            _pending.append(row)
            log.error(f"[DB] rv2_gradings 저장 실패 → 보류 {len(_pending)}건 ({r.get('answer_id')}): "
                      f"{type(e).__name__}: {e}")
            log.error("[DB 보류 행] %s", json.dumps(row, ensure_ascii=False, default=str))
        return
    save_rv2_to_supabase(r)


def save_rv2_to_supabase(r: dict) -> None:
    sb = clients.get_supabase()
    if not sb:
        log.warning("[Supabase] 클라이언트 없음, R v2 저장 스킵")
        return
    row = {k: r.get(k) for k in RV2_SAVE_KEYS + RV2_SAVE_KEYS_V13}
    try:
        # returning="minimal": RLS에서 insert만 허용(select 정책 없음)해도 저장되도록.
        # 기본값(representation)은 저장한 행을 다시 읽어 오므로 select 권한이 없으면 RLS 오류가 난다.
        sb.table("rv2_gradings").insert(row, returning="minimal").execute()
        log.info(f"[Supabase] rv2_gradings 저장 완료 ({r.get('answer_id')})")
        return
    except Exception as e:
        msg = str(e)
        if not any(t in msg for t in ("column", "PGRST204", "schema cache")):
            log.warning(f"[Supabase] rv2_gradings 저장 실패: {e}")
            return
        log.warning("[Supabase] 새 컬럼이 없어 기존 컬럼만 저장합니다 — "
                    "sql/2026-11-24_rv2_gradings_add_columns.sql 실행 필요")
    try:
        sb.table("rv2_gradings").insert({k: r.get(k) for k in RV2_SAVE_KEYS}, returning="minimal").execute()
        log.info(f"[Supabase] rv2_gradings 저장 완료(기존 컬럼만) ({r.get('answer_id')})")
    except Exception as e:
        log.warning(f"[Supabase] rv2_gradings 저장 실패: {e}")


def load_rv2_problems() -> None:
    """엑셀 01_PROBLEMS 문항을 문항 DB에 (다시) 올린다. 서버 시작과 /admin/rv2/reload에서 쓴다."""
    for p in rv2_codebook.problems_for_server_db():
        p["created_at"] = now_iso()
        problems_db[p["id"]] = p


def load_initial_problems() -> None:
    """엑셀 01_PROBLEMS(INV-R-01~10)를 초기 문제로 등록."""
    mats = rv2_codebook.get()
    load_rv2_problems()
    log.info(f"[초기 문제 등록] 엑셀 {len(mats['problems'])}개 문항 로드 "
             f"(R코드 {len(mats['r_codebook'])}개, 기준판정 {len(mats['line_judgment'])}줄, x{mats['xlsx_sha8']})")
