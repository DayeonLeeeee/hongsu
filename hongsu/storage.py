# coding: utf-8
"""저장소: 문항·결과 인메모리 DB, rv2_gradings 저장, 엑셀 문항 로드.

인메모리 DB는 서버 재시작 시 초기화된다. 채점 결과의 영구 저장은 rv2_gradings 테이블:
DATABASE_URL이 있으면 PostgreSQL(Railway, hongsu/db.py), 없으면 Supabase.
"""
import rv2_codebook
from hongsu import clients, db
from hongsu.logging_setup import log
from hongsu.utils import now_iso

problems_db = {}   # id -> problem dict
results_db = {}    # id -> 학생 결과 dict (인메모리, /results)
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
                     "response_models", "extraction", "first_issues"]


def save_rv2(r: dict) -> None:
    """R v2 진단 1건 저장. 응답 원문(raw_responses)까지 저장. 실패해도 서버는 계속 동작."""
    if db.enabled():
        try:
            db.insert_rv2({k: r.get(k) for k in RV2_SAVE_KEYS + RV2_SAVE_KEYS_V13})
            log.info(f"[DB] rv2_gradings 저장 완료 ({r.get('answer_id')})")
        except Exception as e:
            log.warning(f"[DB] rv2_gradings 저장 실패: {type(e).__name__}: {e}")
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
