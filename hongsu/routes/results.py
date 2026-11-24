# coding: utf-8
"""학생 결과 저장·조회(인메모리)와 통계·CSV 내보내기.
영구 저장은 /grade가 rv2_gradings(Postgres 또는 Supabase)에 이미 한다."""
from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

from hongsu import storage
from hongsu.guard import json_body
from hongsu.logging_setup import log
from hongsu.utils import now_iso

bp = Blueprint("results", __name__)


@bp.route("/results", methods=["POST"])
def save_result() -> ResponseReturnValue:
    """채점 결과 저장 (R v2 필드). Supabase 저장은 /grade 단계에서 끝났으므로 여기서는 인메모리만."""
    data = json_body()
    rid = f"R{len(storage.results_db) + 1:06d}"
    result = {
        "result_id": rid,
        "answer_id": data.get("answer_id", ""),
        "problem_id": data.get("problem_id"),
        "student_id": data.get("student_id", "anonymous"),
        "is_correct": data.get("is_correct", data.get("math_result_correct", False)),
        "main_error_code": data.get("main_error_code"),
        "justification_level": data.get("justification_level"),
        "review_needed": data.get("review_needed", False),
        "response_complete": data.get("response_complete"),
        "missing_criteria": data.get("missing_criteria", []),
        "input_status": data.get("input_status", ""),
        "lines": data.get("lines", []),
        "steps": data.get("steps", []),
        "feedback": data.get("feedback", {}),
        "prompt_version": data.get("prompt_version", ""),
        "model": data.get("model", ""),
        "self_check": data.get("self_check", ""),   # 알고 풀었다/헷갈렸다/찍었다
        "created_at": now_iso(),
    }
    storage.results_db[rid] = result
    log.info(f"[결과 저장] {rid}: problem={result['problem_id']}, correct={result['is_correct']}, "
          f"R={result['main_error_code']}, J={result['justification_level']}")
    return jsonify({"success": True, "result_id": rid})


@bp.route("/results", methods=["GET"])
def list_results() -> ResponseReturnValue:
    """결과 목록. ?problem_id=P0001&student_id=xxx 로 필터"""
    pid = request.args.get("problem_id", "")
    sid = request.args.get("student_id", "")
    results = list(storage.results_db.values())
    if pid:
        results = [r for r in results if r.get("problem_id") == pid]
    if sid:
        results = [r for r in results if r.get("student_id") == sid]
    return jsonify({"success": True, "results": results, "total": len(results)})


@bp.route("/results/<rid>", methods=["GET"])
def get_result(rid: str) -> ResponseReturnValue:
    r = storage.results_db.get(rid)
    if not r:
        return jsonify({"success": False, "error": "결과를 찾을 수 없습니다"}), 404
    return jsonify({"success": True, "result": r})


@bp.route("/stats/error_distribution", methods=["GET"])
def error_distribution() -> ResponseReturnValue:
    """전체 오답 유형 분포 통계"""
    from collections import Counter
    codes = [r.get("main_error_code") or "코드없음"
             for r in storage.results_db.values()
             if not r.get("is_correct", True)]
    counter = Counter(codes)
    return jsonify({
        "success": True,
        "total_wrong": sum(counter.values()),
        "distribution": dict(counter),
    })


@bp.route("/results/export", methods=["GET"])
def export_results_csv() -> ResponseReturnValue:
    """채점 결과를 CSV로 내보내기 (답안ID·prompt_version·모델명·시각 포함)"""
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)

    # 헤더
    writer.writerow([
        "result_id", "answer_id", "problem_id", "student_id", "is_correct",
        "main_error_code", "justification_level", "review_needed", "input_status",
        "self_check", "prompt_version", "model", "created_at",
    ])

    for rid, r in storage.results_db.items():
        writer.writerow([
            rid, r.get("answer_id", ""), r.get("problem_id", ""), r.get("student_id", ""),
            r.get("is_correct", ""), r.get("main_error_code", ""), r.get("justification_level", ""),
            r.get("review_needed", ""), r.get("input_status", ""), r.get("self_check", ""),
            r.get("prompt_version", ""), r.get("model", ""), r.get("created_at", ""),
        ])

    csv_data = output.getvalue()
    return csv_data, 200, {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": "attachment; filename=hongsu_results.csv",
    }
