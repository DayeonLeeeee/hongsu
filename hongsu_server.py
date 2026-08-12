"""
suma_server.py  — v2 재설계 서버
─────────────────────────────────
파이프라인:
  1. 문제 OCR:     Mathpix + Gemini Flash 병렬 → 평문 (표시용)
  2. 문제 AI 풀이:  Claude Opus 4.6 (vision + tool_use) → 정답 + 단계별 풀이
  3. 풀이 OCR:     Mathpix + Gemini Flash 병렬 → 단계별 평문 (표시용)
  4. 자연어 수정:  Claude Haiku (vision + tool_use)
  5. 채점:         Claude Opus 4.6 (vision + tool_use)
                    → 13차원 특성 벡터 + observed_errors
                    → 코드로 유클리드 거리 → 최근접 H코드
  6. 개인화 피드백: Claude Haiku (tool_use)
  7. 해설 (마스터): Claude Opus 4.6 (vision + tool_use)

실행:
    pip install flask openai requests anthropic google-generativeai
    python suma_server.py
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
import anthropic
import google.generativeai as genai

# ─── API 키 ──────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

MATHPIX_APP_ID  = os.environ.get("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY", "")

# ─── 모델 이름 ───────────────────────────────────────
MODEL_OPUS   = "claude-opus-4-6"          # 정답, 채점, 해설
MODEL_HAIKU  = "claude-haiku-4-5-20251001" # 수정, 피드백
MODEL_GEMINI = "gemini-2.5-flash"          # OCR 보조

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)


# ─── 공통 유틸 ────────────────────────────────────────
def now_iso() -> str:
    return datetime.datetime.now().isoformat()


def parse_tool_use(response) -> dict:
    """Claude tool_use 응답에서 첫 tool_use 블록의 input dict를 뽑는다"""
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


# ─── 1. Mathpix OCR ──────────────────────────────────
def call_mathpix(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """Mathpix — 수식 특화 OCR. 실패해도 빈 문자열 반환 (병렬용)"""
    try:
        url = "https://api.mathpix.com/v3/text"
        headers = {
            "app_id": MATHPIX_APP_ID,
            "app_key": MATHPIX_APP_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "src": f"data:{mime_type};base64,{image_b64}",
            "formats": ["text", "latex_styled"],
            "data_options": {"include_latex": True},
        }
        resp = http_requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return (data.get("latex_styled") or data.get("text", "")).strip()
    except Exception as e:
        print(f"  [Mathpix 실패] {e}")
    return ""


# ─── 2. Gemini Flash OCR (평문) ──────────────────────
def call_gemini_ocr(image_b64: str, mime_type: str, mode: str = "problem") -> dict:
    """
    Gemini Flash — 평문 OCR (표시 전용).
    LaTeX 백슬래시 없이 사람이 읽기 쉬운 표기로 출력.
    """
    if mode == "solution":
        instruction = (
            "당신은 손글씨 수학 풀이 OCR입니다. "
            "학생이 쓴 풀이를 그대로 옮기세요. 오류를 임의로 고치지 마세요. "
            "단계별로 번호를 나눠 배열로 반환하세요. "
            "표기 규칙 (LaTeX 백슬래시 절대 사용 금지):\n"
            "- 분수: (a/b)\n"
            "- 제곱근: √n 또는 √(...)\n"
            "- 로그 밑: log_2 72 형태 (밑을 _로)\n"
            "- 지수: x^2, 2^n\n"
            "- 첨자: a_n, x_1\n"
            "- 곱셈: ×, 나눗셈: ÷\n"
            "- 부등호: ≤ ≥ ≠\n\n"
            "반드시 JSON으로만 응답:\n"
            "{\"steps\": [{\"index\": 1, \"text\": \"평문 수식\"}, ...]}"
        )
    else:
        instruction = (
            "당신은 수학 문제 OCR입니다. "
            "인쇄된 문제를 그대로 옮기세요. "
            "표기 규칙 (LaTeX 백슬래시 절대 사용 금지):\n"
            "- 분수: (a/b)\n"
            "- 제곱근: √n 또는 √(...)\n"
            "- 로그 밑: log_2 72 형태\n"
            "- 지수: x^2\n"
            "- 첨자: a_n\n"
            "- 곱셈: ×, 나눗셈: ÷\n\n"
            "반드시 JSON으로만 응답:\n"
            '{"text": "평문 문제", "problem_number": "16", "source": "출처"}'
        )

    try:
        model = genai.GenerativeModel(MODEL_GEMINI)
        response = model.generate_content(
            [instruction, {"mime_type": mime_type, "data": base64.b64decode(image_b64)}],
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [Gemini OCR 실패] {e}")
        if mode == "solution":
            return {"steps": []}
        return {"text": "", "problem_number": "", "source": ""}


# ─── 3. 평문 정리 (표기 정규화) ───────────────────────
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


# ═══════════════════════════════════════════════════════
# 오답 유형 프로파일 (H1~H10, 문서 기반)
# ═══════════════════════════════════════════════════════
FEATURE_KEYS = [
    "checked_uniqueness",          # 입력→출력 유일성
    "checked_definition_domain",   # 정의역/공역/치역 구분
    "checked_one_to_one",          # 일대일 판정
    "checked_composition_order",   # 합성 순서
    "checked_domain_restriction",  # 분모≠0, 근호≥0 확인
    "arithmetic_error",            # 산술 실수
    "wrong_formula_applied",       # 잘못된 공식
    "notation_confusion",          # 표기 혼동
    "graph_interpretation_error",  # 그래프 해석
    "has_reasoning",               # 근거 존재
    "used_criterion",              # 판정 기준 명시
    "gave_counterexample",         # 반례 제시
    "final_answer_correct",        # 최종 답 정오
]

ERROR_PROFILES = {
    "H1": {  # 함수 정의 오류
        "checked_uniqueness": 0.1, "checked_definition_domain": 0.5,
        "checked_one_to_one": 0.3, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.5,
        "arithmetic_error": 0.1, "wrong_formula_applied": 0.2,
        "notation_confusion": 0.3, "graph_interpretation_error": 0.1,
        "has_reasoning": 0.4, "used_criterion": 0.2,
        "gave_counterexample": 0.1, "final_answer_correct": 0.3,
    },
    "H2": {  # 정의역·공역·치역 혼동
        "checked_uniqueness": 0.6, "checked_definition_domain": 0.1,
        "checked_one_to_one": 0.4, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.3,
        "arithmetic_error": 0.3, "wrong_formula_applied": 0.5,
        "notation_confusion": 0.7, "graph_interpretation_error": 0.2,
        "has_reasoning": 0.4, "used_criterion": 0.3,
        "gave_counterexample": 0.2, "final_answer_correct": 0.3,
    },
    "H3": {  # 그래프 판정 오류
        "checked_uniqueness": 0.2, "checked_definition_domain": 0.5,
        "checked_one_to_one": 0.3, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.2,
        "arithmetic_error": 0.1, "wrong_formula_applied": 0.2,
        "notation_confusion": 0.3, "graph_interpretation_error": 0.8,
        "has_reasoning": 0.4, "used_criterion": 0.2,
        "gave_counterexample": 0.1, "final_answer_correct": 0.3,
    },
    "H4": {  # 일대일함수·일대일대응 오류
        "checked_uniqueness": 0.5, "checked_definition_domain": 0.5,
        "checked_one_to_one": 0.1, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.2,
        "arithmetic_error": 0.1, "wrong_formula_applied": 0.4,
        "notation_confusion": 0.6, "graph_interpretation_error": 0.2,
        "has_reasoning": 0.5, "used_criterion": 0.3,
        "gave_counterexample": 0.2, "final_answer_correct": 0.4,
    },
    "H5": {  # 합성함수 순서 오류
        "checked_uniqueness": 0.3, "checked_definition_domain": 0.3,
        "checked_one_to_one": 0.1, "checked_composition_order": 0.1,
        "checked_domain_restriction": 0.2,
        "arithmetic_error": 0.3, "wrong_formula_applied": 0.4,
        "notation_confusion": 0.7, "graph_interpretation_error": 0.1,
        "has_reasoning": 0.3, "used_criterion": 0.2,
        "gave_counterexample": 0.1, "final_answer_correct": 0.2,
    },
    "H6": {  # 역함수 존재 조건 오류
        "checked_uniqueness": 0.4, "checked_definition_domain": 0.4,
        "checked_one_to_one": 0.1, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.2,
        "arithmetic_error": 0.2, "wrong_formula_applied": 0.7,
        "notation_confusion": 0.4, "graph_interpretation_error": 0.3,
        "has_reasoning": 0.3, "used_criterion": 0.2,
        "gave_counterexample": 0.1, "final_answer_correct": 0.3,
    },
    "H7": {  # 역함수 식 변형 오류
        "checked_uniqueness": 0.5, "checked_definition_domain": 0.4,
        "checked_one_to_one": 0.6, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.3,
        "arithmetic_error": 0.7, "wrong_formula_applied": 0.5,
        "notation_confusion": 0.8, "graph_interpretation_error": 0.2,
        "has_reasoning": 0.4, "used_criterion": 0.3,
        "gave_counterexample": 0.1, "final_answer_correct": 0.2,
    },
    "H8": {  # 역함수 그래프 오류
        "checked_uniqueness": 0.5, "checked_definition_domain": 0.4,
        "checked_one_to_one": 0.5, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.3,
        "arithmetic_error": 0.2, "wrong_formula_applied": 0.4,
        "notation_confusion": 0.5, "graph_interpretation_error": 0.7,
        "has_reasoning": 0.4, "used_criterion": 0.3,
        "gave_counterexample": 0.2, "final_answer_correct": 0.4,
    },
    "H9": {  # 유리·무리함수 확장 오류
        "checked_uniqueness": 0.5, "checked_definition_domain": 0.6,
        "checked_one_to_one": 0.4, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.1,
        "arithmetic_error": 0.3, "wrong_formula_applied": 0.4,
        "notation_confusion": 0.4, "graph_interpretation_error": 0.6,
        "has_reasoning": 0.4, "used_criterion": 0.3,
        "gave_counterexample": 0.2, "final_answer_correct": 0.3,
    },
    "H10": {  # 근거·정당화 부족
        "checked_uniqueness": 0.5, "checked_definition_domain": 0.5,
        "checked_one_to_one": 0.5, "checked_composition_order": 0.5,
        "checked_domain_restriction": 0.5,
        "arithmetic_error": 0.3, "wrong_formula_applied": 0.3,
        "notation_confusion": 0.3, "graph_interpretation_error": 0.3,
        "has_reasoning": 0.1, "used_criterion": 0.1,
        "gave_counterexample": 0.1, "final_answer_correct": 0.7,
    },
}

ERROR_LABELS = {
    "H1": "함수 정의 오류",
    "H2": "정의역·공역·치역 혼동",
    "H3": "그래프 판정 오류",
    "H4": "일대일함수·일대일대응 오류",
    "H5": "합성함수 순서 오류",
    "H6": "역함수 존재 조건 오류",
    "H7": "역함수 식 변형 오류",
    "H8": "역함수 그래프 오류",
    "H9": "유리·무리함수 확장 오류",
    "H10": "근거·정당화 부족",
}

MISSING_CONCEPTS = {
    "H1": ["대응", "함수의 뜻", "입력-출력의 유일성"],
    "H2": ["정의역", "공역", "치역", "함숫값"],
    "H3": ["함수의 그래프", "같은 x값", "수직선 판정"],
    "H4": ["일대일함수", "일대일대응", "공역=치역 조건"],
    "H5": ["합성함수", "안쪽 함수", "대응 순서"],
    "H6": ["역함수", "일대일대응", "정의역 제한"],
    "H7": ["역함수 구하기", "식 변형", "정의역·치역 교환"],
    "H8": ["y=x 대칭", "좌표 교환", "역함수 그래프"],
    "H9": ["정의역 제한", "치역", "점근선", "그래프 변환"],
    "H10": ["수학적 근거 쓰기", "조건 확인", "자기 점검"],
}

FEEDBACK_TEMPLATES = {
    "H1": "함수인지 볼 때는 모든 x를 한꺼번에 보지 말고, x값 하나를 잡아 그때 y값이 하나만 정해지는지 확인해 보세요.",
    "H2": "공역 전체가 항상 치역은 아닙니다. 정의역의 값을 하나씩 넣어 실제로 나오는 y값만 다시 모아 보세요.",
    "H3": "그래프가 함수인지 볼 때는 곡선이 하나인지보다, 같은 x값에서 y값이 하나만 나오는지가 더 중요합니다.",
    "H4": "서로 다른 x가 서로 다른 y로 가면 일대일함수입니다. 여기에 공역의 모든 원소가 실제로 사용되면 일대일대응입니다.",
    "H5": "합성함수는 안쪽 함수가 먼저입니다. 입력값이 먼저 g를 지나고, 그 결과가 f로 들어가는 흐름을 화살표로 그려 보세요.",
    "H6": "역함수는 식을 거꾸로 푸는 것만으로 끝나지 않습니다. 먼저 원래 함수가 일대일로 대응하는지 확인해야 합니다.",
    "H7": "역함수를 구할 때는 먼저 x를 y로 나타내고, 그 다음 x와 y를 바꿉니다. 두 단계를 한 번에 처리하면 실수가 자주 납니다.",
    "H8": "역함수의 그래프는 원래 그래프의 점 (a,b)를 (b,a)로 바꾼 점들의 모임입니다. 그래서 y=x에 대하여 대칭이 됩니다.",
    "H9": "유리함수와 무리함수는 그래프를 그리기 전에 정의역 제한을 먼저 확인해야 합니다.",
    "H10": "답만 쓰면 어떤 생각으로 풀었는지 알기 어렵습니다. 사용한 조건을 한 문장으로 덧붙여 보세요.",
}


# ═══════════════════════════════════════════════════════
# 벡터 분류 유틸
# ═══════════════════════════════════════════════════════
def euclidean(v1: dict, v2: dict) -> float:
    return math.sqrt(sum((v1.get(k, 0.5) - v2.get(k, 0.5)) ** 2 for k in FEATURE_KEYS))


def classify_error(student_vector: dict) -> dict:
    """13차원 벡터 → 각 H와의 거리 + 최근접 유형 + 분포"""
    distances = {h: euclidean(student_vector, p) for h, p in ERROR_PROFILES.items()}
    sorted_h = sorted(distances.items(), key=lambda x: x[1])

    primary = sorted_h[0][0]
    primary_d = sorted_h[0][1]
    secondary = sorted_h[1][0]
    secondary_d = sorted_h[1][1]
    gap = secondary_d - primary_d

    # 분포 (거리 → 유사도 → 정규화)
    similarities = {h: 1.0 / (1.0 + d) for h, d in distances.items()}
    total = sum(similarities.values())
    distribution = {h: round(s / total, 4) for h, s in similarities.items()}

    return {
        "primary_h": primary,
        "primary_label": ERROR_LABELS[primary],
        "primary_distance": round(primary_d, 4),
        "secondary_h": secondary,
        "secondary_label": ERROR_LABELS[secondary],
        "secondary_distance": round(secondary_d, 4),
        "gap": round(gap, 4),
        "is_typical": primary_d < 1.0,
        "distribution": distribution,
        "all_distances": {h: round(d, 4) for h, d in distances.items()},
    }


# ═══════════════════════════════════════════════════════
# Tool 정의 (tool_use 강제)
# ═══════════════════════════════════════════════════════

# ─── (1) 문제 → 정답/풀이 ─────────────────────────────
TOOL_SOLVE_PROBLEM = {
    "name": "solve_problem",
    "description": "수학 문제를 스스로 풀어 정답과 단계별 풀이를 반환합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "최종 답 (간단한 텍스트, 예: 5, 65, 3/5)",
            },
            "difficulty": {"type": "string", "enum": ["하", "중", "상"]},
            "method_type": {"type": "string", "description": "풀이 방법 이름"},
            "concept_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "사용 개념 태그 2~3개",
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "title": {"type": "string", "description": "10자 이내"},
                        "content": {"type": "string", "description": "1~2문장, 수식은 $...$"},
                        "formula": {"type": "string", "description": "핵심 식 한 줄 $...$"},
                    },
                    "required": ["step", "title", "content", "formula"],
                },
                "description": "난이도 하: 2~3단계, 중: 3~4단계, 상: 4~5단계",
            },
            "verification": {"type": "string", "description": "답을 검산한 한 줄"},
        },
        "required": ["answer", "difficulty", "method_type", "concept_tags", "steps", "verification"],
    },
}

# ─── (2) 자연어 수정 (문제/풀이 공용) ─────────────────
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

# ─── (3) 채점 (특성 벡터 추출) ────────────────────────
TOOL_GRADE_SOLUTION = {
    "name": "grade_student_solution",
    "description": "학생 풀이의 오류 특성을 13차원 벡터로 추출하고 단계별 판정합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_correct": {"type": "boolean", "description": "최종 답이 정답과 일치하는가"},
            "student_final_answer": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "text": {"type": "string", "description": "학생이 쓴 내용 (평문)"},
                        "status": {"type": "string", "enum": ["ok", "wrong", "depends"]},
                        "explanation": {"type": "string", "description": "이 단계에 대한 짧은 설명"},
                    },
                    "required": ["index", "text", "status", "explanation"],
                },
            },
            "error_step_index": {
                "type": ["integer", "null"],
                "description": "첫 오류 단계 번호 (정답이면 null)",
            },
            "features": {
                "type": "object",
                "properties": {
                    "checked_uniqueness": {"type": "number"},
                    "checked_definition_domain": {"type": "number"},
                    "checked_one_to_one": {"type": "number"},
                    "checked_composition_order": {"type": "number"},
                    "checked_domain_restriction": {"type": "number"},
                    "arithmetic_error": {"type": "number"},
                    "wrong_formula_applied": {"type": "number"},
                    "notation_confusion": {"type": "number"},
                    "graph_interpretation_error": {"type": "number"},
                    "has_reasoning": {"type": "number"},
                    "used_criterion": {"type": "number"},
                    "gave_counterexample": {"type": "number"},
                    "final_answer_correct": {"type": "number"},
                },
                "required": [
                    "checked_uniqueness", "checked_definition_domain", "checked_one_to_one",
                    "checked_composition_order", "checked_domain_restriction",
                    "arithmetic_error", "wrong_formula_applied", "notation_confusion",
                    "graph_interpretation_error", "has_reasoning", "used_criterion",
                    "gave_counterexample", "final_answer_correct",
                ],
                "description": (
                    "13차원 오류 특성 벡터. 각 값 0.0~1.0. "
                    "1.0=완벽/명확히 확인, 0.5=애매/해당없음, 0.0=전혀 없음/심각한 오류"
                ),
            },
            "observed_errors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "관찰된 구체적 실수 목록",
            },
        },
        "required": [
            "is_correct", "student_final_answer", "steps",
            "error_step_index", "features", "observed_errors",
        ],
    },
}

# ─── (4) 개인화 피드백 ────────────────────────────────
TOOL_PERSONAL_FEEDBACK = {
    "name": "compose_feedback",
    "description": "학생 실수와 오답 유형을 종합해 짧고 안내적인 피드백을 생성합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "학생에게 줄 피드백 (2문장 이내, 안내 어조)",
            },
            "next_focus": {
                "type": "array",
                "items": {"type": "string"},
                "description": "다음에 학생이 확인/연습할 것 1~2개",
            },
        },
        "required": ["feedback", "next_focus"],
    },
}


# ═══════════════════════════════════════════════════════
# 엔드포인트 1: OCR (문제/풀이 공용)
# ═══════════════════════════════════════════════════════
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    Mathpix + Gemini Flash 병렬 실행 → 평문 텍스트 반환.
    표시 전용. 채점/풀이 생성은 이미지 원본을 별도로 사용.
    """
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        mime_type = data.get("mime_type", "image/jpeg")
        mode = data.get("mode", "problem")   # "problem" | "solution"

        print(f"\n[OCR 요청] mode={mode}, 이미지={len(image_b64)//1024}KB")

        # 병렬 실행
        f_mathpix = executor.submit(call_mathpix, image_b64, mime_type)
        f_gemini  = executor.submit(call_gemini_ocr, image_b64, mime_type, mode)
        mathpix_result = f_mathpix.result()
        gemini_result  = f_gemini.result()

        print(f"  → Mathpix: {mathpix_result[:60]}")
        print(f"  → Gemini: {json.dumps(gemini_result, ensure_ascii=False)[:100]}")

        if mode == "solution":
            steps = gemini_result.get("steps", [])
            # 각 단계 평문 정규화
            steps = [
                {"index": s.get("index", i + 1), "text": normalize_plain(s.get("text", ""))}
                for i, s in enumerate(steps)
            ]
            return jsonify({
                "success": True,
                "mode": "solution",
                "steps": steps,
                "mathpix_raw": mathpix_result,
                "confidence": 0.9 if steps else 0.3,
                "timestamp": now_iso(),
            })
        else:
            final_text = normalize_plain(gemini_result.get("text", ""))
            return jsonify({
                "success": True,
                "mode": "problem",
                "final_text": final_text,
                "problem_number": gemini_result.get("problem_number", ""),
                "source": gemini_result.get("source", ""),
                "mathpix_raw": mathpix_result,
                "confidence": 0.9 if final_text else 0.3,
                "timestamp": now_iso(),
            })

    except Exception as e:
        print(f"  [OCR 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════
# 엔드포인트 2: 문제 → AI 풀이/정답
# ═══════════════════════════════════════════════════════
@app.route("/solve", methods=["POST"])
def solve_endpoint():
    """
    Claude Opus vision — 문제 이미지에서 직접 정답과 단계별 풀이 생성.
    /solution 대신 사용 (해설 생성).
    """
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        mime_type = data.get("mime_type", "image/jpeg")
        problem_text = data.get("problem_text", "")   # 참고자료 (있으면)

        print(f"\n[문제 풀이 요청] 이미지={len(image_b64)//1024}KB, 참고 텍스트={len(problem_text)}자")

        prompt = f"""당신은 고등학교 수학 모범 풀이 전문가입니다.

첨부된 문제 이미지를 보고 스스로 정확히 풀어 정답과 단계별 풀이를 작성하세요.

[참고 텍스트 (OCR 결과, 이미지가 우선)]
{problem_text if problem_text else "(없음)"}

[작성 규칙]
1. 먼저 문제를 직접 풀어 정답을 확인하고 검산하세요.
2. 정석적이고 교과서적인 풀이로 작성. 여러 풀이법을 나열하지 마세요.
3. 단계 수는 난이도에 맞춰:
   - "하": 2~3단계
   - "중": 3~4단계
   - "상": 4~5단계
4. title 10자 이내, content 1~2문장, formula 한 줄.
5. 수식은 $...$ 로 감싸고 한글은 $ 밖에.
6. answer는 $ 없이 간단한 텍스트로.

반드시 solve_problem tool로만 응답하세요."""

        response = claude_client.messages.create(
            model=MODEL_OPUS,
            max_tokens=4096,
            temperature=0.0,
            tools=[TOOL_SOLVE_PROBLEM],
            tool_choice={"type": "tool", "name": "solve_problem"},
            system="정확한 수학 문제 해결자. 답을 검산하는 습관이 있으며 tool_use로만 응답합니다.",
            messages=[{
                "role": "user",
                "content": [
                    make_image_block(image_b64, mime_type),
                    {"type": "text", "text": prompt},
                ]
            }],
        )
        result = parse_tool_use(response)
        result["success"] = True
        result["timestamp"] = now_iso()

        print(f"  [풀이 완료] 정답={result.get('answer')}, 난이도={result.get('difficulty')}, "
              f"단계={len(result.get('steps', []))}")
        return jsonify(result)

    except Exception as e:
        print(f"  [풀이 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════
# 엔드포인트 3: 자연어 수정 (문제/풀이 공용)
# ═══════════════════════════════════════════════════════
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

        prompt = f"""당신은 수학 OCR 편집 도우미입니다.

[현재 OCR 결과]
{current_text}

[사용자 수정 지시]
{user_instruction}

[작업]
1. 사용자 지시를 이해하고 현재 텍스트를 수정하세요.
2. 원본 이미지를 참고하여 사용자 지적이 이미지와 일치하는지 확인.
3. 이미지와 다르면 사용자 지적을 우선하되, note 필드로 안내.
4. 평문 표기 유지 (LaTeX 백슬래시 사용 금지).

반드시 refine_ocr_text tool로만 응답하세요."""

        response = claude_client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=1024,
            temperature=0.0,
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


# ═══════════════════════════════════════════════════════
# 엔드포인트 4: 채점 (특성 벡터 + 최근접 분류)
# ═══════════════════════════════════════════════════════
@app.route("/grade", methods=["POST"])
def grade_endpoint():
    """
    Claude Opus vision — 문제/풀이 이미지 직접 보고 채점.
    → 13차원 특성 벡터 뽑음
    → 코드로 최근접 H코드 결정
    → Claude Haiku로 개인화 피드백 생성

    입력:
      problem_image_b64: 문제 이미지 (필수)
      solution_image_b64: 학생 풀이 이미지 (필수)
      problem_text: OCR 평문 (참고용, 선택)
      solution_steps: [{index, text}] OCR 평문 단계별 (참고용, 선택)
      correct_answer: 정답 (있으면 전달, 없으면 서버가 판단)
      grading_rules: 채점 룰 텍스트 (교수님 문서 기반, 선택)
    """
    try:
        data = request.get_json()
        problem_image = data.get("problem_image_b64", "")
        solution_image = data.get("solution_image_b64", "")
        problem_text = data.get("problem_text", "")
        solution_steps = data.get("solution_steps", [])
        correct_answer = data.get("correct_answer", "")
        grading_rules = data.get("grading_rules", "")

        if not problem_image or not solution_image:
            return jsonify({"success": False, "error": "problem_image_b64와 solution_image_b64 필수"}), 400

        print(f"\n[채점 요청] 문제 이미지={len(problem_image)//1024}KB, "
              f"풀이 이미지={len(solution_image)//1024}KB, 참고 단계={len(solution_steps)}개")

        # 참고 텍스트 조립
        steps_str = "\n".join([f"  Step {s.get('index')}: {s.get('text')}" for s in solution_steps])
        if not steps_str:
            steps_str = "(없음)"

        prompt = f"""당신은 고등학교 수학 채점 전문가입니다.

두 이미지를 보고 학생 풀이를 채점하세요:
- 첫 번째 이미지: 원본 문제
- 두 번째 이미지: 학생의 손글씨 풀이

[참고 - OCR 평문 결과 (이미지가 항상 우선)]
문제 텍스트: {problem_text if problem_text else "(없음)"}
학생 풀이 단계:
{steps_str}

[알려진 정답 (있으면)]
{correct_answer if correct_answer else "(없음, 스스로 풀어 확인)"}

[채점 규칙]
{grading_rules if grading_rules else '''
1. 문제를 직접 풀어 정답을 스스로 구한 뒤 학생 풀이를 채점하세요.
2. 학생이 다른 방법을 써도 논리적으로 타당하면 ok.
3. 표기 실수는 관대하게, 실제 수학적 오류만 wrong으로.
4. 최초 오류 단계만 wrong, 그 이후는 depends.
5. 최종 답이 맞고 논리에 큰 결함 없으면 is_correct=true.
'''}

[특성 벡터 작성 지침 - 매우 중요]
아래 13개 특성을 각각 0.0~1.0 사이 값으로 매기세요.
학생 풀이 이미지에서 관찰된 실제 근거로만 판단하세요.

- checked_uniqueness: 입력→출력 유일성 확인. 함수 관련 문제 아니면 0.5.
- checked_definition_domain: 정의역/공역/치역 구분 사용. 관련 없으면 0.5.
- checked_one_to_one: 일대일 판정 시도. 관련 없으면 0.5.
- checked_composition_order: 합성함수 순서 인식. 관련 없으면 0.5.
- checked_domain_restriction: 분모≠0, 근호≥0 등 정의역 제한 확인. 관련 없으면 0.5.
- arithmetic_error: 산술 실수 정도. 1.0=심각, 0.0=완벽.
- wrong_formula_applied: 잘못된 공식 사용 정도.
- notation_confusion: 표기 혼동 정도.
- graph_interpretation_error: 그래프 해석/그리기 오류. 관련 없으면 0.0.
- has_reasoning: 근거 문장 존재 여부. 1.0=상세, 0.0=답만.
- used_criterion: 판정 기준 명시. 1.0=명확, 0.0=없음.
- gave_counterexample: 반례/구체 예시 제시.
- final_answer_correct: 최종 답 정오. 1.0=정답, 0.0=오답.

observed_errors: 이미지에서 관찰한 구체적 실수를 짧은 문장으로 나열.

반드시 grade_student_solution tool로만 응답하세요."""

        response = claude_client.messages.create(
            model=MODEL_OPUS,
            max_tokens=4096,
            temperature=0.0,
            tools=[TOOL_GRADE_SOLUTION],
            tool_choice={"type": "tool", "name": "grade_student_solution"},
            system="수학 채점 전문가. 이미지 근거에 기반해 특성 벡터를 정확히 추출합니다.",
            messages=[{
                "role": "user",
                "content": [
                    make_image_block(problem_image, "image/jpeg"),
                    make_image_block(solution_image, "image/jpeg"),
                    {"type": "text", "text": prompt},
                ]
            }],
        )
        grading = parse_tool_use(response)

        # 특성 벡터 → 최근접 H코드
        features = grading.get("features", {})
        classification = classify_error(features)

        # 개인화 피드백 생성 (Claude Haiku)
        feedback = generate_feedback(
            h_code=classification["primary_h"],
            observed_errors=grading.get("observed_errors", []),
            student_context=steps_str,
        )

        result = {
            "success": True,
            "is_correct": grading.get("is_correct", False),
            "student_final_answer": grading.get("student_final_answer", ""),
            "error_step_index": grading.get("error_step_index"),
            "steps": grading.get("steps", []),
            "features": features,
            "observed_errors": grading.get("observed_errors", []),
            "classification": classification,
            "feedback": feedback,
            "timestamp": now_iso(),
        }

        print(f"  [채점 완료] correct={result['is_correct']}, "
              f"primary={classification['primary_h']} ({classification['primary_label']}), "
              f"gap={classification['gap']}, typical={classification['is_typical']}")
        return jsonify(result)

    except Exception as e:
        print(f"  [채점 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


def generate_feedback(h_code: str, observed_errors: list, student_context: str) -> dict:
    """Claude Haiku로 개인화 피드백 생성"""
    try:
        template = FEEDBACK_TEMPLATES.get(h_code, "")
        concepts = MISSING_CONCEPTS.get(h_code, [])
        label = ERROR_LABELS.get(h_code, h_code)

        prompt = f"""학생에게 줄 짧은 피드백을 작성하세요.

[진단된 오답 유형]
{h_code}: {label}

[표준 피드백 예시 (참고)]
{template}

[누락 개념]
{', '.join(concepts)}

[이 학생의 구체적 실수]
{chr(10).join('- ' + e for e in observed_errors) if observed_errors else "(관찰 없음)"}

[학생 풀이 요약]
{student_context}

[규칙]
1. 학생 풀이의 구체적 증거를 언급하며 시작.
2. 빠진 개념을 한두 개로 좁혀서 짚기.
3. 다음에 무엇을 확인할지 안내.
4. 두 문장 이내로 짧게.
5. "틀렸다"보다 "다음에 어떻게 하면 되는지" 어조.

반드시 compose_feedback tool로만 응답하세요."""

        response = claude_client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=512,
            temperature=0.3,
            tools=[TOOL_PERSONAL_FEEDBACK],
            tool_choice={"type": "tool", "name": "compose_feedback"},
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_tool_use(response)
    except Exception as e:
        print(f"  [피드백 생성 실패] {e}")
        return {
            "feedback": FEEDBACK_TEMPLATES.get(h_code, "다시 한번 풀이 근거를 확인해 보세요."),
            "next_focus": MISSING_CONCEPTS.get(h_code, [])[:2],
        }


# ═══════════════════════════════════════════════════════
# 엔드포인트 5: 마스터 해설 (문제 이미지 → 상세 해설)
# ═══════════════════════════════════════════════════════
@app.route("/master_solution", methods=["POST"])
def master_solution_endpoint():
    """
    문제 이미지를 vision으로 읽어 마스터 해설 생성.
    내부적으로 /solve와 같은 tool을 재사용.
    """
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        mime_type = data.get("mime_type", "image/jpeg")
        problem_text = data.get("problem_text", "")

        print(f"\n[마스터 해설 요청] 이미지={len(image_b64)//1024}KB")

        prompt = f"""당신은 고등학교 수학 해설 전문가입니다.

첨부된 문제 이미지를 보고 학생용 마스터 해설을 작성하세요.

[참고 텍스트 (OCR, 이미지가 우선)]
{problem_text if problem_text else "(없음)"}

[해설 원칙]
1. 정답을 스스로 검산.
2. 가장 정석적인 하나의 풀이만 제시. 다른 풀이법 나열 금지.
3. 단계 수: 난이도 하 2~3, 중 3~4, 상 4~5.
4. title 10자 이내, content 1~2문장, formula 한 줄.
5. 학생이 이해할 수 있게 명확히.

반드시 solve_problem tool로만 응답하세요."""

        response = claude_client.messages.create(
            model=MODEL_OPUS,
            max_tokens=4096,
            temperature=0.0,
            tools=[TOOL_SOLVE_PROBLEM],
            tool_choice={"type": "tool", "name": "solve_problem"},
            system="수학 해설 전문가. 검산 후 정석 풀이 하나만 tool_use로 반환.",
            messages=[{
                "role": "user",
                "content": [
                    make_image_block(image_b64, mime_type),
                    {"type": "text", "text": prompt},
                ]
            }],
        )
        result = parse_tool_use(response)
        result["success"] = True
        result["timestamp"] = now_iso()

        print(f"  [해설 완료] 정답={result.get('answer')}, "
              f"난이도={result.get('difficulty')}, 단계={len(result.get('steps', []))}")
        return jsonify(result)

    except Exception as e:
        print(f"  [해설 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════
# 엔드포인트 6: 헬스체크
# ═══════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": now_iso(), "version": "v2"})


# ═══════════════════════════════════════════════════════
# 관리자 웹 UI (브라우저에서 /admin 접속)
# ═══════════════════════════════════════════════════════

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
        os.path.join(os.getc
