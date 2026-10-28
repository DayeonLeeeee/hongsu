"""
hongsu 서버 (R v2)

파이프라인:
  1. 풀이 OCR:     Mathpix + Gemini Flash 병렬 → 단계별 평문 (/ocr)
  2. 자연어 수정:  Claude Haiku (vision + tool_use) (/refine) → 학생이 OCR 결과 확인
  3. R v2 진단:    확인된 OCR 텍스트 → 줄별 ok/wrong/depends + R코드 + J수준 (/grade)
                  키트(kit-1.2-20261022) 로직 이식. 문항·코드북·기준판정·규칙은 전부
                   엑셀(materials/dataset_rv2.xlsx)에서 조립, 문항별 허용 코드만 노출,
                   인용 원문 대조·허용 밖 코드 거부·1회 재시도, 기권(review_needed),
                   응답 원문·재현성 필드 저장 (Supabase rv2_gradings)
  4. 피드백:       다음 단계 (엑셀 08시트 질문→힌트→해설). 현재는 빈 값
  5. 모범답안      (/master_solution)

hongsu-rv2-1.1: 기존 H코드 체계(H1~H10 상수·분류·H 피드백·100점 루브릭 채점, /grade_h) 제거.
                문항은 엑셀 01_PROBLEMS(INV-R-01~10)만 사용.
"""

import os
import json
import base64
import re
import math
import asyncio
import datetime
import traceback
import statistics
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import requests as http_requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

MATHPIX_APP_ID  = os.environ.get("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# 진단 모델은 rv2_grader.RV2_MODEL (환경변수 RV2_MODEL, 기본 claude-opus-4-6). 아래는 OCR·수정용.
MODEL_HAIKU  = "claude-haiku-4-5-20251001"
MODEL_GEMINI = "gemini-2.5-flash"

# 프롬프트 분리 모듈
import prompts as PROMPTS
from prompts import ocr as p_ocr
from prompts import refine as p_refine
from prompts import rv2 as p_rv2      # R v2 프롬프트 (키트 prompt_builder 이식본)
import rv2_codebook                   # 엑셀 로더 (코드북·문항·기준판정·규칙)
import rv2_grader                     # R v2 진단 실행·검증·재시도

# 반복 오답 임계값 (미확정, 임의): 이 횟수 넘게 오답이면 정답 공개
MAX_ATTEMPTS_BEFORE_REVEAL = 3

claude_client = None
_gemini_configured = False

def get_claude():
    global claude_client
    if claude_client is None:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return claude_client

def get_gemini_model():
    global _gemini_configured
    import google.generativeai as genai
    if not _gemini_configured:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_configured = True
    return genai.GenerativeModel(MODEL_GEMINI)


_supabase = None
def get_supabase():
    global _supabase
    if _supabase is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("  [Supabase] 연결 성공")
        except Exception as e:
            print(f"  [Supabase] 연결 실패: {e}")
    return _supabase

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)


# ─── 공통 유틸 ─────
def now_iso() -> str:
    return datetime.datetime.now().isoformat()


def parse_tool_use(response) -> dict:
    """Claude tool_use 응답에서 첫 tool_use 블록의 input dict를 뽑아내기"""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    # tool_use가 없으면 text에서 JSON 파싱 시도
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text = block.text.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            try:
                return json.loads(text)
            except Exception:
                pass
    raise ValueError("Claude 응답에서 구조화된 결과를 찾지 못함")


def make_image_block(image_b64: str, mime_type: str = "image/jpeg") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": image_b64,
        }
    }


# ─── 1. Mathpix OCR ────
def call_mathpix(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """
    Mathpix — 수식 특화 OCR. 실패해도 빈 문자열 반환 (병렬용).

    옵션 설계:
    - formats: ["text"]만 요청 (latex_styled는 백슬래시 반환하는데
      프로젝트 규칙상 평문 표기라서 후처리 부담. text 필드가 이미
      $...$ 안에 수식이 있는 평문이라 normalize_plain으로 정리 가능).
    - math_inline_delimiters: ["$", "$"] — 수식 경계 명확히
    - rm_spaces: false — 손글씨 간격 정보 유지 (분수 등 구분)
    - numbers_default_to_math: true — 손글씨 숫자를 텍스트가 아닌 수식으로
    - include_line_data: false — 라인 단위 분할은 Gemini가 담당
    """
    if not (MATHPIX_APP_ID and MATHPIX_APP_KEY):
        return ""
    try:
        url = "https://api.mathpix.com/v3/text"
        headers = {
            "app_id": MATHPIX_APP_ID,
            "app_key": MATHPIX_APP_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "src": f"data:{mime_type};base64,{image_b64}",
            "formats": ["text"],
            "math_inline_delimiters": ["$", "$"],
            "rm_spaces": False,
            "numbers_default_to_math": True,
        }
        resp = http_requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            text = (data.get("text", "") or "").strip()
            # 신뢰도 로그 (있으면)
            conf = data.get("confidence")
            if conf is not None:
                print(f"  [Mathpix] confidence={conf:.2f}, {len(text)}자")
            return text
        else:
            print(f"  [Mathpix 실패] HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        print(f"  [Mathpix 실패] {e}")
    return ""


# ─── 2. Gemini Flash OCR (평문) ─────
def call_gemini_ocr(image_b64: str, mime_type: str) -> dict:
    """
    Gemini Flash — 학생 풀이 평문 OCR.
    문제는 관리자가 텍스트로 입력하므로 OCR 대상 아님.
    """
    instruction = p_ocr.build_solution_ocr_prompt()
    try:
        model = get_gemini_model()
        response = model.generate_content(
            [instruction, {"mime_type": mime_type, "data": base64.b64decode(image_b64)}],
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [Gemini OCR 실패] {e}")
        return {"steps": []}


# ─── 3. 평문 정리 (표기 정규화) ─────
def normalize_plain(text: str) -> str:
    """Gemini가 어쩌다 LaTeX 백슬래시를 뱉으면 평문으로 강제 변환"""
    if not text:
        return ""
    s = text
    # LaTeX \frac → (a/b)
    s = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1/\2)', s)
    # \sqrt{n} → √n
    s = re.sub(r'\\sqrt\s*\{([^{}]+)\}', r'√\1', s)
    # \log_{n} → log_n
    s = re.sub(r'\\log_\s*\{([^{}]+)\}', r'log_\1', s)
    s = re.sub(r'\\log', 'log', s)
    s = re.sub(r'\\ln', 'ln', s)
    # 삼각함수
    for f in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']:
        s = re.sub(rf'\\{f}\b', f, s)
    # 그리스 문자
    greek = {'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ',
             'delta': 'δ', 'lambda': 'λ', 'pi': 'π', 'infty': '∞'}
    for k, v in greek.items():
        s = re.sub(rf'\\{k}\b', v, s)
    # 연산자
    s = re.sub(r'\\cdot', '·', s)
    s = re.sub(r'\\times', '×', s)
    s = re.sub(r'\\div', '÷', s)
    s = re.sub(r'\\leq', '≤', s)
    s = re.sub(r'\\geq', '≥', s)
    s = re.sub(r'\\neq', '≠', s)
    s = re.sub(r'\\pm', '±', s)
    # 괄호 잔재
    s = re.sub(r'\\left', '', s)
    s = re.sub(r'\\right', '', s)
    # $ 제거
    s = s.replace('$', '')
    # 남은 명령어
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    # 중괄호 정리
    s = re.sub(r'\{([^{}]+)\}', r'\1', s)
    s = s.replace('{', '').replace('}', '')
    # 공백 정리
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Tool 정의 (tool_use 강제)

# ─── (1) 자연어 수정 (문제/풀이 공용) ────
TOOL_REFINE_TEXT = {
    "name": "refine_ocr_text",
    "description": "사용자의 자연어 수정 지시를 반영해 OCR 평문을 갱신합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "corrected_text": {"type": "string", "description": "수정된 최종 평문"},
            "changes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "구체적 변경 내역 (예: 'log5 → log_5')",
            },
            "image_agreement": {
                "type": "string",
                "enum": ["matches", "disagrees", "unclear"],
                "description": "학생 수정이 원본 이미지와 일치하는지",
            },
            "note": {"type": "string", "description": "이미지와 불일치 시 안내, 없으면 빈 문자열"},
        },
        "required": ["corrected_text", "changes", "image_agreement", "note"],
    },
}

# 엔드포인트 1: OCR
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    학생 풀이 OCR: Mathpix + Gemini Flash 병렬 실행 → 단계별 평문 텍스트 반환.
    문제는 관리자가 텍스트로 등록하므로 OCR 대상 아님.
    """
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        mime_type = data.get("mime_type", "image/jpeg")

        print(f"\n[OCR 요청] 이미지={len(image_b64)//1024}KB")

        # 병렬 실행
        f_mathpix = executor.submit(call_mathpix, image_b64, mime_type)
        f_gemini  = executor.submit(call_gemini_ocr, image_b64, mime_type)
        mathpix_result = f_mathpix.result()
        gemini_result  = f_gemini.result()

        print(f"  → Mathpix: {mathpix_result[:60]}")
        print(f"  → Gemini: {json.dumps(gemini_result, ensure_ascii=False)[:100]}")

        steps = gemini_result.get("steps", [])
        steps = [
            {"index": s.get("index", i + 1), "text": normalize_plain(s.get("text", ""))}
            for i, s in enumerate(steps)
        ]
        return jsonify({
            "success": True,
            "steps": steps,
            "mathpix_raw": mathpix_result,
            "timestamp": now_iso(),
        })

    except Exception as e:
        print(f"  [OCR 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        print(f"  [OCR 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# 엔드포인트 2: 자연어 수정
@app.route("/refine", methods=["POST"])
def refine_endpoint():
    """
    사용자 자연어 지시로 OCR 평문 수정.
    풀이 단계 지정 수정도 여기서 처리 ("2단계에서 3+4로").
    """
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        mime_type = data.get("mime_type", "image/jpeg")
        current_text = data.get("current_text", "")
        user_instruction = data.get("instruction", "")

        print(f"\n[수정 요청] instruction={user_instruction[:60]}")

        prompt = p_refine.build(current_text=current_text, user_instruction=user_instruction)

        response = get_claude().messages.create(
            model=MODEL_HAIKU,
            max_tokens=1024,
            tools=[TOOL_REFINE_TEXT],
            tool_choice={"type": "tool", "name": "refine_ocr_text"},
            system="정확한 텍스트 편집자. 자연어 지시를 정확히 반영합니다.",
            messages=[{
                "role": "user",
                "content": [
                    make_image_block(image_b64, mime_type),
                    {"type": "text", "text": prompt},
                ]
            }],
        )
        result = parse_tool_use(response)
        result["corrected_text"] = normalize_plain(result.get("corrected_text", ""))
        result["success"] = True
        result["timestamp"] = now_iso()

        print(f"  [수정 완료] changes={result.get('changes')}, agreement={result.get('image_agreement')}")
        return jsonify(result)

    except Exception as e:
        print(f"  [수정 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# 엔드포인트 3-R: R v2 진단 (메인)
@app.route("/grade", methods=["POST"])
def grade_rv2_endpoint():
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
        data = request.get_json() or {}
        problem_id = data.get("problem_id", "")
        mode = data.get("mode", "with_codebook")
        input_status = data.get("input_status", "confirmed")
        attempt_count = int(data.get("attempt_count", 1))
        student_id = data.get("student_id", "anonymous")

        if mode not in ("with_codebook", "grade_only"):
            return jsonify({"success": False, "error": f"mode 값 오류: {mode}"}), 400
        if not rv2_codebook.has_problem(problem_id):
            return jsonify({"success": False,
                            "error": f"R v2 문항이 아님: {problem_id} (엑셀 01_PROBLEMS에 없음)"}), 400

        answer_text = data.get("solution_text", "") or ""
        if data.get("solution_steps"):
            answer_text = rv2_grader.steps_to_answer_text(data["solution_steps"])
        if not answer_text.strip() and data.get("solution_image_b64"):
            return jsonify({"success": False,
                            "error": "R v2는 확인된 OCR 텍스트로 채점합니다. /ocr → (/refine) 후 solution_steps로 보내세요."}), 400

        answer_id = f"A-{now_iso().replace(':','').replace('-','').replace('.','')[:15]}-{problem_id}"
        print(f"\n[R v2 진단] {answer_id} mode={mode} status={input_status}")

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
                              "math_result_correct": None, "review_needed": False, "review_reason": ""}}
        else:
            rec = rv2_grader.diagnose(get_claude(), problem_id, answer_text, mode=mode)

        res = rec.get("result") or {}
        review_needed = bool(res.get("review_needed")) or bool(rec.get("validation_issues")) or bool(rec.get("error"))
        if "max_tokens" in (rec.get("stop_reasons") or []):
            review_needed = True

        result = {
            "success": rec.get("error") is None,
            "answer_id": answer_id,
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
            "review_needed": review_needed,
            "review_reason": res.get("review_reason", ""),
            "validation_issues": rec.get("validation_issues", []),
            "error": rec.get("error"),
            # ── 재현성 필드 (README 규칙 5) ──
            "prompt_version": rec.get("prompt_version"),
            "codebook_version": rec.get("codebook_version"),
            "xlsx_sha8": rec.get("xlsx_sha8"),
            "allowed_codes": rec.get("allowed_codes"),
            "model": rec.get("model"),
            "timestamp": now_iso(),
            "n_calls": rec.get("n_calls"),
            "usage": rec.get("usage"),
            "stop_reasons": rec.get("stop_reasons"),
            # ── 기존 프론트 호환 ──
            "is_correct": res.get("math_result_correct"),
            "steps": rv2_grader.to_server_steps(res),
            "reveal_answer": (res.get("math_result_correct") is False) and attempt_count >= MAX_ATTEMPTS_BEFORE_REVEAL,
            "feedback": {"feedback": "", "detailed_feedback": "", "next_focus": []},  # 피드백은 다음 단계
        }

        # 응답 원문 포함 전체 저장 (원문은 응답 JSON에는 싣지 않음 — 용량)
        save_rv2_to_supabase({**result, "answer_text": rec.get("answer_text"),
                              "raw_responses": rec.get("raw_responses")})

        print(f"  [R v2 완료] main={result['main_error_code']} J={result['justification_level']} "
              f"review={review_needed} calls={rec.get('n_calls')} issues={rec.get('validation_issues')}")
        status = 200 if result["success"] else 502
        return jsonify(result), status

    except Exception as e:
        print(f"  [R v2 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/admin/rv2/reload", methods=["POST"])
def rv2_reload():
    """엑셀을 다시 읽는다 (서버 재시작 없이 코드북·문항 갱신)."""
    try:
        mats = rv2_codebook.reload()
        for p in rv2_codebook.problems_for_server_db():
            p["created_at"] = now_iso()
            _problems_db[p["id"]] = p
        return jsonify({"success": True, "xlsx_sha8": mats["xlsx_sha8"],
                        "problems": len(mats["problems"]), "r_codes": len(mats["r_codebook"])})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 엔드포인트 5: 모범답안 해설 (DB 조회)
@app.route("/master_solution", methods=["POST"])
def master_solution_endpoint():
    """DB에 저장된 모범답안(model_answer)을 반환."""
    try:
        data = request.get_json()
        problem_id = data.get("problem_id", "")
        problem_text = data.get("problem_text", "")

        prob_data = _problems_db.get(problem_id, {})

        if not prob_data and problem_text:
            for p in _problems_db.values():
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

    except Exception as e:
        print(f"  [해설 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500



# 엔드포인트 6: 헬스체크

@app.route("/health", methods=["GET"])
def health():
    m = rv2_codebook.get()
    return jsonify({"status": "ok", "time": now_iso(), "version": "hongsu-rv2-1.1",
                    "rv2_model": rv2_grader.RV2_MODEL, "xlsx_sha8": m.get("xlsx_sha8"),
                    "rv2_problems": len(m.get("problems", []))})


# 관리자 웹 UI (브라우저에서 /admin 접속)

@app.route("/admin")
def admin_page():
    """관리자 웹 페이지 — 문제 관리, 결과 조회, 통계"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin.html"),
        os.path.join(os.getcwd(), "admin.html"),
        "/app/admin.html",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return f"<h1>admin.html not found</h1><p>Tried: {candidates}</p>", 404


@app.route("/student")
def student_page():
    """학생용 웹앱"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "student.html"),
        os.path.join(os.getcwd(), "student.html"),
        "/app/student.html",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return f"<h1>student.html not found</h1>", 404


@app.route("/manifest.json")
def manifest():
    """PWA manifest"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"),
        os.path.join(os.getcwd(), "manifest.json"),
        "/app/manifest.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "application/json"}


# 엔드포인트 7: 일관성 측정
@app.route("/measure_consistency", methods=["POST"])
def measure_consistency_endpoint():
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
        data = request.get_json()
        n_runs = min(int(data.get("n_runs", 3)), 5)

        runs = []
        for i in range(n_runs):
            print(f"\n[일관성 측정 {i+1}/{n_runs}]")
            # 채점 호출 (내부적으로 grade_endpoint 로직 재사용)
            # 여기서는 간단히 요청을 다시 보냄
            with app.test_client() as c:
                resp = c.post("/grade", json=data)
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

    except Exception as e:
        print(f"  [일관성 측정 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500



# 관리 API: 문제 & 채점 가이드라인 CRUD


# 인메모리 저장소 (서버 재시작하면 초기화. Supabase가 영구 저장)
_problems_db = {}   # id -> problem dict
_results_db = {}    # id -> grading result dict
_next_id = 1


# ─── Supabase 저장: R v2 진단 (rv2_gradings 테이블, sql/rv2_gradings.sql) ───
def save_rv2_to_supabase(r: dict):
    """R v2 진단 1건 저장. 응답 원문(raw_responses)까지 저장. 실패해도 서버는 계속 동작."""
    sb = get_supabase()
    if not sb:
        print("  [Supabase] 클라이언트 없음, R v2 저장 스킵")
        return
    keys = ["answer_id", "problem_id", "student_id", "attempt_count", "mode", "input_status",
            "answer_text", "lines", "main_error_code", "justification_level", "math_result_correct",
            "review_needed", "review_reason", "validation_issues", "error",
            "prompt_version", "codebook_version", "xlsx_sha8", "allowed_codes", "model",
            "n_calls", "usage", "stop_reasons", "raw_responses"]
    row = {k: r.get(k) for k in keys}
    try:
        # returning="minimal": RLS에서 insert만 허용(select 정책 없음)해도 저장되도록.
        # 기본값(representation)은 저장한 행을 다시 읽어 오므로 select 권한이 없으면 RLS 오류가 난다.
        sb.table("rv2_gradings").insert(row, returning="minimal").execute()
        print(f"  [Supabase] rv2_gradings 저장 완료 ({r.get('answer_id')})")
    except Exception as e:
        print(f"  [Supabase] rv2_gradings 저장 실패: {e}")


def _gen_id():
    global _next_id
    pid = f"P{_next_id:04d}"
    _next_id += 1
    return pid


@app.route("/admin/problems", methods=["GET"])
def list_problems():
    """문제 목록 조회. ?unit=함수 로 필터 가능."""
    unit = request.args.get("unit", "")
    problems = list(_problems_db.values())
    if unit:
        problems = [p for p in problems if p.get("unit") == unit]
    return jsonify({"success": True, "problems": problems, "total": len(problems)})


@app.route("/admin/problems", methods=["POST"])
def add_problem():
    """문제 추가. 외부에서 JSON으로 문제 + 채점 가이드 입력."""
    data = request.get_json()
    pid = data.get("id") or _gen_id()

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
    _problems_db[pid] = problem
    print(f"  [문제 추가] {pid}: {problem['title']}")
    return jsonify({"success": True, "id": pid, "problem": problem})


@app.route("/admin/problems/<pid>", methods=["GET"])
def get_problem(pid):
    """문제 상세 조회"""
    p = _problems_db.get(pid)
    if not p:
        return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404
    return jsonify({"success": True, "problem": p})


@app.route("/admin/problems/<pid>", methods=["PUT"])
def update_problem(pid):
    """문제 수정 (부분 업데이트)"""
    if pid not in _problems_db:
        return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404
    data = request.get_json()
    for key, val in data.items():
        if key != "id":   # id는 변경 불가
            _problems_db[pid][key] = val
    _problems_db[pid]["updated_at"] = now_iso()
    print(f"  [문제 수정] {pid}: {list(data.keys())}")
    return jsonify({"success": True, "problem": _problems_db[pid]})


@app.route("/admin/problems/<pid>", methods=["DELETE"])
def delete_problem(pid):
    """문제 삭제"""
    if pid not in _problems_db:
        return jsonify({"success": False, "error": "문제를 찾을 수 없습니다"}), 404
    del _problems_db[pid]
    return jsonify({"success": True})


@app.route("/admin/problems/bulk", methods=["POST"])
def bulk_add_problems():
    """문제 일괄 등록 (배열로 보내기)"""
    data = request.get_json()
    problems = data.get("problems", [])
    added = []
    for p in problems:
        pid = p.get("id") or _gen_id()
        p["id"] = pid
        p["created_at"] = now_iso()
        _problems_db[pid] = p
        added.append(pid)
    print(f"  [일괄 등록] {len(added)}개 문제")
    return jsonify({"success": True, "added_ids": added, "total": len(_problems_db)})


# 학생 결과 저장/조회 API (인메모리. 영구 저장은 /grade가 Supabase rv2_gradings에 이미 함)
@app.route("/results", methods=["POST"])
def save_result():
    """채점 결과 저장 (R v2 필드). Supabase 저장은 /grade 단계에서 끝났으므로 여기서는 인메모리만."""
    data = request.get_json() or {}
    rid = f"R{len(_results_db) + 1:06d}"
    result = {
        "result_id": rid,
        "answer_id": data.get("answer_id", ""),
        "problem_id": data.get("problem_id"),
        "student_id": data.get("student_id", "anonymous"),
        "is_correct": data.get("is_correct", data.get("math_result_correct", False)),
        "main_error_code": data.get("main_error_code"),
        "justification_level": data.get("justification_level"),
        "review_needed": data.get("review_needed", False),
        "input_status": data.get("input_status", ""),
        "lines": data.get("lines", []),
        "steps": data.get("steps", []),
        "feedback": data.get("feedback", {}),
        "prompt_version": data.get("prompt_version", ""),
        "model": data.get("model", ""),
        "self_check": data.get("self_check", ""),   # 알고 풀었다/헷갈렸다/찍었다
        "created_at": now_iso(),
    }
    _results_db[rid] = result
    print(f"  [결과 저장] {rid}: problem={result['problem_id']}, correct={result['is_correct']}, "
          f"R={result['main_error_code']}, J={result['justification_level']}")
    return jsonify({"success": True, "result_id": rid})


@app.route("/results", methods=["GET"])
def list_results():
    """결과 목록. ?problem_id=P0001&student_id=xxx 로 필터"""
    pid = request.args.get("problem_id", "")
    sid = request.args.get("student_id", "")
    results = list(_results_db.values())
    if pid:
        results = [r for r in results if r.get("problem_id") == pid]
    if sid:
        results = [r for r in results if r.get("student_id") == sid]
    return jsonify({"success": True, "results": results, "total": len(results)})


@app.route("/results/<rid>", methods=["GET"])
def get_result(rid):
    r = _results_db.get(rid)
    if not r:
        return jsonify({"success": False, "error": "결과를 찾을 수 없습니다"}), 404
    return jsonify({"success": True, "result": r})


@app.route("/stats/error_distribution", methods=["GET"])
def error_distribution():
    """전체 오답 유형 분포 통계"""
    from collections import Counter
    codes = [r.get("main_error_code") or "코드없음"
             for r in _results_db.values()
             if not r.get("is_correct", True)]
    counter = Counter(codes)
    return jsonify({
        "success": True,
        "total_wrong": sum(counter.values()),
        "distribution": dict(counter),
    })


@app.route("/results/export", methods=["GET"])
def export_results_csv():
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

    for rid, r in _results_db.items():
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



# 교수님 문서 기반 문제 초기 등록 (서버 시작 시)

def load_initial_problems():
    """엑셀 01_PROBLEMS(INV-R-01~10)를 초기 문제로 등록.
    (기존 Q01~Q10 하드코딩은 제거 — 원본은 hongsu_server.orig.py 참고)"""
    mats = rv2_codebook.get()
    for p in rv2_codebook.problems_for_server_db():
        p["created_at"] = now_iso()
        _problems_db[p["id"]] = p
    print(f"  [초기 문제 등록] 엑셀 {len(mats['problems'])}개 문항 로드 "
          f"(R코드 {len(mats['r_codebook'])}개, 기준판정 {len(mats['line_judgment'])}줄, x{mats['xlsx_sha8']})")


# ─── 초기 문제 로드  ───
load_initial_problems()
# Supabase problems 테이블 병합은 제거 (문항은 엑셀 01_PROBLEMS만 사용)


# ─── 서버 시작 (로컬 실행용) ────
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("=" * 60)
    print("  hongsu v2 서버 (vision + tool_use)")
    print(f"  로컬: http://localhost:5000")
    print(f"  네트워크: http://{local_ip}:5000")
    print(f"  학생 앱: http://{local_ip}:5000/student")
    print(f"  관리자:  http://{local_ip}:5000/admin")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
