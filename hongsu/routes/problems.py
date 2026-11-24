# coding: utf-8
"""문항 관리(/admin/problems, /admin/rv2/reload)와 모범답안 조회(/master_solution)."""
from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

import rv2_codebook
from hongsu import storage
from hongsu.guard import BadInput, json_body
from hongsu.logging_setup import log
from hongsu.utils import now_iso

bp = Blueprint("problems", __name__)


@bp.route("/admin/problems", methods=["GET"])
def list_problems() -> ResponseReturnValue:
    """문제 목록 조회. ?unit=함수 로 필터 가능."""
    unit = request.args.get("unit", "")
    problems = list(storage.problems_db.values())
    if unit:
        problems = [p for p in problems if p.get("unit") == unit]
    return jsonify({"success": True, "problems": problems, "total": len(problems)})


@bp.route("/admin/problems", methods=["POST"])
def add_problem() -> ResponseReturnValue:
    """문제 추가. 외부에서 JSON으로 문제 + 채점 가이드 입력."""
    data = json_body()
    pid = data.get("id") or storage.gen_id()

    problem = {
        "id": pid,
        "unit": data.get("unit", "함수"),                  # 함수/역함수/합성함수/유리함수/무리함수
        "type": data.get("type", "서술형"),                 # 서술형/객관식
        "title": data.get("title", ""),
        "statement": data.get("statement", ""),              # 문제 지문
        "choices": data.get("choices"),                      # 객관식 보기 (없으면 null)
        "answer_index": data.get("answer_index"),            # 객관식 정답 인덱스
        "correct_answer": data.get("correct_answer", ""),    # 서술형 정답
        "problem_image_url": data.get("problem_image_url", ""),

        # 채점 가이드라인 (교수님 문서 기반)
        "grading_rules": data.get("grading_rules", ""),             # 전체 채점 룰 텍스트
        "required_reasoning": data.get("required_reasoning", ""),   # 필수 풀이 근거
        "observation_points": data.get("observation_points", ""),   # 관찰 지점
        "deduction_scale": data.get("deduction_scale", {}),          # 감점 척도 (교수님 제공, 4기준별 설명)
        "rubric": data.get("rubric", []),                           # 부분점수 루브릭 [{기준, 서술 기준}, ...]
        "expected_errors": data.get("expected_errors", []),         # 예상 오류 H코드 목록
        "model_answer": data.get("model_answer", ""),               # 모범답안 텍스트
        "feedback_example": data.get("feedback_example", ""),       # 피드백 예시

        # 메타
        "concept_tags": data.get("concept_tags", []),
        "chapter": data.get("chapter", "함수와 그래프"),
        "score": data.get("score", 100),                     # 배점
        "created_at": now_iso(),
    }
    storage.problems_db[pid] = problem
    log.info(f"[문제 추가] {pid}: {problem['title']}")
    return jsonify({"success": True, "id": pid, "problem": problem})


@bp.route("/admin/problems/<pid>", methods=["GET"])
def get_problem(pid: str) -> ResponseReturnValue:
    """문제 상세 조회"""
    p = storage.problems_db.get(pid)
    if not p:
        return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404
    return jsonify({"success": True, "problem": p})


@bp.route("/admin/problems/<pid>", methods=["PUT"])
def update_problem(pid: str) -> ResponseReturnValue:
    """문제 수정 (부분 업데이트)"""
    if pid not in storage.problems_db:
        return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404
    data = json_body()
    for key, val in data.items():
        if key != "id":   # id는 변경 불가
            storage.problems_db[pid][key] = val
    storage.problems_db[pid]["updated_at"] = now_iso()
    log.info(f"[문제 수정] {pid}: {list(data.keys())}")
    return jsonify({"success": True, "problem": storage.problems_db[pid]})


@bp.route("/admin/problems/<pid>", methods=["DELETE"])
def delete_problem(pid: str) -> ResponseReturnValue:
    """문제 삭제"""
    if pid not in storage.problems_db:
        return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404
    del storage.problems_db[pid]
    return jsonify({"success": True})


@bp.route("/admin/problems/bulk", methods=["POST"])
def bulk_add_problems() -> ResponseReturnValue:
    """문제 일괄 등록 (배열로 보내기)"""
    data = json_body()
    problems = data.get("problems", [])
    if not isinstance(problems, list) or not all(isinstance(p, dict) for p in problems):
        raise BadInput("problems: 문항 객체 배열이어야 함")
    added = []
    for p in problems:
        pid = p.get("id") or storage.gen_id()
        p["id"] = pid
        p["created_at"] = now_iso()
        storage.problems_db[pid] = p
        added.append(pid)
    log.info(f"[일괄 등록] {len(added)}개 문제")
    return jsonify({"success": True, "added_ids": added, "total": len(storage.problems_db)})


@bp.route("/admin/rv2/reload", methods=["POST"])
def rv2_reload() -> ResponseReturnValue:
    """엑셀을 다시 읽는다 (서버 재시작 없이 코드북·문항 갱신)."""
    try:
        mats = rv2_codebook.reload()
        storage.load_rv2_problems()
        return jsonify({"success": True, "xlsx_sha8": mats["xlsx_sha8"],
                        "problems": len(mats["problems"]), "r_codes": len(mats["r_codebook"])})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/master_solution", methods=["POST"])
def master_solution_endpoint() -> ResponseReturnValue:
    """DB에 저장된 모범답안(model_answer)을 반환."""
    try:
        data = json_body()
        problem_id = data.get("problem_id", "")
        problem_text = data.get("problem_text", "")

        prob_data = storage.problems_db.get(problem_id, {})

        if not prob_data and problem_text:
            for p in storage.problems_db.values():
                if problem_text.strip() in p.get("statement", ""):
                    prob_data = p
                    break

        if not prob_data:
            return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404

        model_answer = prob_data.get("model_answer", "")
        if not model_answer:
            return jsonify({"success": False, "error": "모범답안이 등록되지 않았습니다"}), 404

        return jsonify({
            "success": True,
            "problem_id": prob_data.get("id", problem_id),
            "title": prob_data.get("title", ""),
            "correct_answer": prob_data.get("correct_answer", ""),
            "model_answer": model_answer,
            "required_reasoning": prob_data.get("required_reasoning", ""),
            "concept_tags": prob_data.get("concept_tags", []),
            "timestamp": now_iso(),
        })

    except BadInput:
        raise
    except Exception as e:
        log.exception("[해설 에러]")
        return jsonify({"success": False, "error": str(e)}), 500
