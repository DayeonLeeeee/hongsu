# coding: utf-8
"""R v2 진단(/grade)과 반복 일관성 측정(/measure_consistency)."""
import uuid

from flask import Blueprint, current_app, jsonify
from flask.typing import ResponseReturnValue

import rv2_codebook
import rv2_grader
import usage_cost
from hongsu import clients, storage
from hongsu.config import MAX_ANSWER_CHARS, MAX_ATTEMPTS_BEFORE_REVEAL, MAX_STEPS
from hongsu.guard import INTERNAL_FLAG, BadInput, get_int, get_str, json_body, rate_limited
from hongsu.logging_setup import current_request_id, log
from hongsu.utils import now_iso
from prompts import rv2 as p_rv2

bp = Blueprint("grade", __name__)


@bp.route("/grade", methods=["POST"])
@rate_limited
def grade_rv2_endpoint() -> ResponseReturnValue:
    """
    R v2 진단 — 이용오 선생님 키트(run_eval/prompt_builder) 로직의 서버 이식.

    입력 (JSON):
      problem_id      : 엑셀 01_PROBLEMS의 문항 ID (예: INV-R-01) — 필수
      solution_steps  : [{index, text}] 학생이 확인한 OCR 단계 (권장)
      solution_text   : 학생 답안 텍스트 (steps가 없을 때, 합성답안 실험용)
      input_status    : confirmed(기본) | not_attempted | unreadable  (엑셀 02c 시트)
      mode            : with_codebook(기본, 3단계) | grade_only(코드북 없이, 2단계 절제 실험용)
      student_id, attempt_count : 선택
    이미지만 보내면 400 — R v2는 학생/교사가 확인한 OCR 텍스트(input_status=confirmed)로 판정한다.
    """
    try:
        data = json_body()
        problem_id = get_str(data, "problem_id", max_len=50)
        mode = get_str(data, "mode", "with_codebook", max_len=30)
        input_status = get_str(data, "input_status", "confirmed", max_len=30)
        attempt_count = get_int(data, "attempt_count", 1, 1, 1000)
        student_id = get_str(data, "student_id", "anonymous", max_len=100) or "anonymous"
        if input_status not in ("confirmed", "not_attempted", "unreadable"):
            raise BadInput(f"input_status 값 오류: {input_status}")

        if mode not in ("with_codebook", "grade_only"):
            return jsonify({"success": False, "error": f"mode 값 오류: {mode}"}), 400
        if not rv2_codebook.has_problem(problem_id):
            return jsonify({"success": False,
                            "error": f"R v2 문항이 아님: {problem_id} (엑셀 01_PROBLEMS에 없음)"}), 400

        answer_text = get_str(data, "solution_text", max_len=MAX_ANSWER_CHARS)
        steps = data.get("solution_steps")
        if steps:
            if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
                raise BadInput("solution_steps: [{index, text}, …] 형식이어야 함")
            if len(steps) > MAX_STEPS:
                raise BadInput(f"solution_steps: 너무 많음 ({len(steps)}개, 최대 {MAX_STEPS}개)")
            for s in steps:
                get_str(s, "text", max_len=MAX_ANSWER_CHARS)
            answer_text = rv2_grader.steps_to_answer_text(steps)
            if len(answer_text) > MAX_ANSWER_CHARS:
                raise BadInput(f"답안이 너무 김 ({len(answer_text)}자, 최대 {MAX_ANSWER_CHARS}자)")
        if not answer_text.strip() and data.get("solution_image_b64"):
            return jsonify({"success": False,
                            "error": "R v2는 확인된 OCR 텍스트로 채점합니다. /ocr → (/refine) 후 solution_steps로 보내세요."}), 400

        # 시각(초 단위)만으로는 같은 문항을 같은 초에 낸 학생끼리 ID가 겹친다 → 뒤에 무작위 6자
        answer_id = (f"A-{now_iso().replace(':','').replace('-','').replace('.','')[:15]}-{problem_id}"
                     f"-{uuid.uuid4().hex[:6]}")
        log.info(f"[R v2 진단] {answer_id} mode={mode} status={input_status}")

        # 백지·판독 불가는 오류가 아니다 (판정 규칙 2) → LLM 호출 없이 기록
        if input_status in ("not_attempted", "unreadable") or not answer_text.strip():
            rec = {"problem_id": problem_id, "mode": mode, "model": None,
                   "prompt_version": p_rv2.build_prompt(rv2_codebook.get(), problem_id, mode)["version"],
                   "codebook_version": rv2_codebook.get().get("codebook_version"),
                   "xlsx_sha8": rv2_codebook.get().get("xlsx_sha8"),
                   "allowed_codes": rv2_codebook.allowed_codes(problem_id), "answer_text": answer_text,
                   "ts": now_iso(), "raw_responses": [], "usage": {}, "n_calls": 0, "stop_reasons": [],
                   "validation_issues": [], "error": None,
                   "result": {"lines": [], "input_status": input_status if answer_text.strip() else "not_attempted",
                              "main_error_code": None, "justification_level": None,
                              "math_result_correct": None, "review_needed": False, "review_reason": "",
                              "response_complete": None, "missing_criteria": []}}
        else:
            rec = rv2_grader.diagnose(clients.get_claude(), problem_id, answer_text, mode=mode)

        res = rec.get("result") or {}
        outcome, review_sources = rv2_grader.classify_outcome(rec)
        review_needed = outcome in ("review", "error")

        result = {
            "success": rec.get("error") is None,
            "answer_id": answer_id,
            "request_id": current_request_id(),   # 서버 로그 검색용
            "problem_id": problem_id,
            "student_id": student_id,
            "attempt_count": attempt_count,
            "mode": mode,
            # ── R v2 진단 결과 ──
            "input_status": res.get("input_status", input_status),
            "lines": res.get("lines", []),
            "main_error_code": res.get("main_error_code"),
            "justification_level": res.get("justification_level"),
            "math_result_correct": res.get("math_result_correct"),
            "outcome": outcome,                  # graded / review / incomplete / not_attempted / unreadable / error
            "review_sources": review_sources,    # review로 간 이유: model(기권) / validation / max_tokens / error
            "review_needed": review_needed,
            "review_reason": res.get("review_reason", ""),
            "response_complete": res.get("response_complete"),       # R-v3 규칙 12: False면 미완 답안
            "completion_status": {True: "complete", False: "incomplete"}.get(res.get("response_complete")),
            "missing_criteria": res.get("missing_criteria", []),      # 대응 풀이가 없는 기준 번호 (코드 아님, 피드백용)
            "validation_issues": rec.get("validation_issues", []),
            "error": rec.get("error"),
            # ── 재현성 필드 (README 규칙 5) ──
            "prompt_version": rec.get("prompt_version"),
            "grader_version": rec.get("grader_version", rv2_grader.GRADER_VERSION),
            "git_commit": rec.get("git_commit", rv2_grader.code_commit()),
            "codebook_version": rec.get("codebook_version"),
            "xlsx_sha8": rec.get("xlsx_sha8"),
            "allowed_codes": rec.get("allowed_codes"),
            "model": rec.get("model"),
            "response_models": rec.get("response_models", []),
            "timestamp": now_iso(),
            "n_calls": rec.get("n_calls"),
            "usage": rec.get("usage"),
            "cost_usd": rec.get("cost_usd", 0.0),           # 이 답안 채점 비용 (USD, 재시도 포함)
            "stop_reasons": rec.get("stop_reasons"),
            "tool_choice": rec.get("tool_choice"),
            "extraction": rec.get("extraction", []),        # 호출별 결과 출처 tool/text/none (스키마 안정성 측정)
            "first_issues": rec.get("first_issues", []),    # 첫 호출 정제·검증 사유 (재시도로 해결돼도 기록)
            # ── 기존 프론트 호환 ──
            "is_correct": res.get("math_result_correct"),
            "steps": rv2_grader.to_server_steps(res),
            "reveal_answer": (res.get("math_result_correct") is False) and attempt_count >= MAX_ATTEMPTS_BEFORE_REVEAL,
            "feedback": {"feedback": "", "detailed_feedback": "", "next_focus": []},  # 피드백은 다음 단계
        }

        # 응답 원문 포함 전체 저장 (원문은 응답 JSON에는 싣지 않음 — 용량)
        storage.save_rv2({**result, "answer_text": rec.get("answer_text"),
                          "raw_responses": rec.get("raw_responses")})

        log.info(f"[R v2 완료] outcome={outcome} sources={review_sources} main={result['main_error_code']} J={result['justification_level']} "
              f"complete={result['response_complete']} missing={result['missing_criteria']} "
              f"review={review_needed} calls={rec.get('n_calls')} issues={rec.get('validation_issues')} "
              f"tokens={(rec.get('usage') or {}).get('input_tokens', 0)}/{(rec.get('usage') or {}).get('output_tokens', 0)} "
              f"cost=${result['cost_usd'] if result['cost_usd'] is not None else '?'}")
        if rec.get("n_calls"):
            usage_cost.add("grade", rec.get("model"), rec.get("usage"), result["cost_usd"], rec.get("n_calls") or 0)
        status = 200 if result["success"] else 502
        return jsonify(result), status

    except BadInput:
        raise
    except Exception as e:
        log.exception("[R v2 에러]")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/measure_consistency", methods=["POST"])
@rate_limited
def measure_consistency_endpoint() -> ResponseReturnValue:
    """
    같은 입력으로 채점을 여러 번 실행해서 재현 일관성을 측정.
    연구 목적 (Cohen's κ 등).

    입력:
      solution_image_b64: 학생 풀이 이미지 (이미지 또는 텍스트 중 하나)
      solution_text: 학생 풀이 텍스트 (합성답안 실험용)
      problem_id: DB 문제 ID
      n_runs: 반복 횟수 (기본 3, 최대 5)
    """
    try:
        data = json_body()
        n_runs = min(get_int(data, "n_runs", 3, 1, 1000), 5)   # 최대 5회 (비용)

        runs = []
        for i in range(n_runs):
            log.info(f"[일관성 측정 {i+1}/{n_runs}]")
            # 채점 호출 (내부적으로 grade_endpoint 로직 재사용)
            # 여기서는 간단히 요청을 다시 보냄
            with current_app.test_client() as c:
                resp = c.post("/grade", json=data, environ_base={INTERNAL_FLAG: True})  # 횟수 제한 제외
                runs.append(resp.get_json())

        # 통계 계산 (R v2: main_error_code·J 기준. 코드 없음(None)도 하나의 판정으로 셈)
        from collections import Counter
        ok_runs = [r for r in runs if r.get("success")]
        counter = Counter(str(r.get("main_error_code")) for r in ok_runs)
        j_counter = Counter(str(r.get("justification_level")) for r in ok_runs)
        most_common, count = counter.most_common(1)[0] if counter else ("N/A", 0)
        j_count = j_counter.most_common(1)[0][1] if j_counter else 0

        return jsonify({
            "success": True,
            "n_runs": n_runs,
            "code_consistency": round(count / n_runs, 4) if n_runs else 0,
            "j_consistency": round(j_count / n_runs, 4) if n_runs else 0,
            "primary_distribution": dict(counter),
            "j_distribution": dict(j_counter),
            "runs": runs,
            "timestamp": now_iso(),
        })

    except BadInput:
        raise
    except Exception as e:
        log.exception("[일관성 측정 에러]")
        return jsonify({"success": False, "error": str(e)}), 500
