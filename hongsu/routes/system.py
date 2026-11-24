# coding: utf-8
"""상태 점검(/health)과 정적 페이지(/admin, /student, /manifest.json)."""
import os

from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

import rv2_codebook
import rv2_grader
from hongsu import clients, db
from hongsu.config import ANTHROPIC_API_KEY, BASE_DIR
from hongsu.utils import now_iso
from prompts import rv2 as p_rv2

bp = Blueprint("system", __name__)


@bp.route("/health", methods=["GET"])
def health() -> ResponseReturnValue:
    """기본: 서버·코드북 상태만 (가벼움, 상태 점검용).
    ?deep=1: 결과 DB(Postgres 또는 Supabase) 조회와 Claude API(모델 조회, 토큰 비용 없음)까지 확인. 하나라도 실패하면 503."""
    m = rv2_codebook.get()
    body = {"status": "ok", "time": now_iso(), "version": p_rv2.PORT_VERSION,
            "grader_version": rv2_grader.GRADER_VERSION, "git_commit": rv2_grader.code_commit(),
            "rv2_model": rv2_grader.RV2_MODEL, "tool_choice": rv2_grader.RV2_TOOL_CHOICE,
            "temperature": rv2_grader.RV2_TEMPERATURE,
            "xlsx_sha8": m.get("xlsx_sha8"), "codebook_version": m.get("codebook_version"),
            "rv2_problems": len(m.get("problems", []))}
    if request.args.get("deep") not in ("1", "true", "yes"):
        return jsonify(body)

    checks = {}
    sb = None if db.enabled() else clients.get_supabase()
    if db.enabled():   # DATABASE_URL(Railway Postgres)에 저장 중
        checks["db"] = db.check()
    elif sb is None:
        checks["supabase"] = {"ok": False, "error": "클라이언트 없음 (SUPABASE_URL/KEY 확인)"}
    else:
        try:
            sb.table("rv2_gradings").select("answer_id").limit(1).execute()
            checks["supabase"] = {"ok": True}
        except Exception as e:
            checks["supabase"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    if not ANTHROPIC_API_KEY:
        checks["claude"] = {"ok": False, "error": "ANTHROPIC_API_KEY 없음"}
    else:
        try:
            info = clients.get_claude().models.retrieve(rv2_grader.RV2_MODEL)
            checks["claude"] = {"ok": True, "model": getattr(info, "id", rv2_grader.RV2_MODEL)}
        except Exception as e:
            checks["claude"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    body["checks"] = checks
    if not all(c["ok"] for c in checks.values()):
        body["status"] = "degraded"
        return jsonify(body), 503
    return jsonify(body)


def _serve_file(name: str, content_type: str) -> ResponseReturnValue:
    """저장소 루트의 파일을 그대로 돌려준다. 실행 위치가 달라도 찾도록 후보 경로를 차례로 본다."""
    candidates = [os.path.join(BASE_DIR, name), os.path.join(os.getcwd(), name), f"/app/{name}"]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": content_type}
    return f"<h1>{name} not found</h1><p>Tried: {candidates}</p>", 404


@bp.route("/admin")
def admin_page() -> ResponseReturnValue:
    """관리자 웹 페이지 — 문제 관리, 결과 조회, 통계"""
    return _serve_file("admin.html", "text/html; charset=utf-8")


@bp.route("/student")
def student_page() -> ResponseReturnValue:
    """학생용 웹앱"""
    return _serve_file("student.html", "text/html; charset=utf-8")


@bp.route("/manifest.json")
def manifest() -> ResponseReturnValue:
    """PWA manifest"""
    return _serve_file("manifest.json", "application/json")
