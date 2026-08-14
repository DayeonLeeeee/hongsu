"""
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

# ─── API 키 ──────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

MATHPIX_APP_ID  = os.environ.get("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ─── 모델 이름 ───────────────────────────────────────
MODEL_OPUS   = "claude-opus-4-6"
MODEL_HAIKU  = "claude-haiku-4-5-20251001"
MODEL_GEMINI = "gemini-2.5-flash"

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

# Supabase 클라이언트 (지연 로드)
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
        model = get_gemini_model()
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


# 단원별 관련 H코드 (해당 단원에서 주로 발생하는 오류)
UNIT_H_RELEVANCE = {
    "함수":     ["H1", "H2", "H3", "H10"],
    "역함수":   ["H4", "H6", "H7", "H8", "H10"],
    "합성함수": ["H5", "H2", "H10"],
    "유리함수": ["H9", "H2", "H10"],
    "무리함수": ["H9", "H2", "H10"],
}


def classify_error(student_vector: dict, unit: str = "") -> dict:
    """13차원 벡터 → 각 H와의 거리 + 최근접 유형 + 분포
    unit이 주어지면 해당 단원과 무관한 H코드에 페널티를 줘서 분포를 명확히 함.
    """
    relevant = UNIT_H_RELEVANCE.get(unit, [])
    print(f"  [분류] unit='{unit}', relevant={relevant}")

    distances = {}
    for h, profile in ERROR_PROFILES.items():
        d = euclidean(student_vector, profile)
        # 해당 단원과 무관한 H코드는 거리 5배 페널티
        if relevant and h not in relevant:
            d *= 5.0
        distances[h] = d

    sorted_h = sorted(distances.items(), key=lambda x: x[1])

    primary = sorted_h[0][0]
    primary_d = sorted_h[0][1]
    secondary = sorted_h[1][0]
    secondary_d = sorted_h[1][1]
    gap = secondary_d - primary_d

    # 분포 (거리의 역수 제곱 → 차이를 더 크게)
    similarities = {h: 1.0 / (1.0 + d) ** 2 for h, d in distances.items()}
    total = sum(similarities.values())
    distribution = {h: round(s / total, 4) for h, s in similarities.items()}

    print(f"  [분류] primary={primary}({round(primary_d,3)}), secondary={secondary}({round(secondary_d,3)}), gap={round(gap,3)}")

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
    "description": "학생 풀이를 채점합니다: 13차원 특성 벡터 + 루브릭별 부분점수 + 단계별 판정.",
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
            "rubric_scores": {
                "type": "array",
                "description": "루브릭 기준별 부분점수. 프롬프트에 주어진 루브릭 순서대로.",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string", "description": "기준명 (예: 개념 사용)"},
                        "points": {"type": "integer", "description": "부여한 점수 (0 ~ max_points)"},
                        "max_points": {"type": "integer", "description": "이 기준의 배점"},
                        "reason": {"type": "string", "description": "왜 이 점수인지 근거 (감점 사유 또는 만점 이유)"},
                        "evidence": {"type": "string", "description": "학생 풀이에서 뽑은 증거 인용 (없으면 빈 문자열)"},
                    },
                    "required": ["criterion", "points", "max_points", "reason", "evidence"],
                },
            },
            "total_score": {
                "type": "integer",
                "description": "루브릭 점수 합산 (전체 배점 중 몇 점)",
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
                "description": "13차원 오류 특성 벡터. 각 값 0.0~1.0.",
            },
            "observed_errors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "관찰된 구체적 실수 목록",
            },
        },
        "required": [
            "is_correct", "student_final_answer", "steps",
            "rubric_scores", "total_score",
            "features", "observed_errors",
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

        response = get_claude().messages.create(
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

        response = get_claude().messages.create(
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
        problem_id = data.get("problem_id", "")
        unit = data.get("unit", "")

        # 문제 ID로 추가 정보 조회
        prob_data = _problems_db.get(problem_id, {})
        if not unit:
            unit = prob_data.get("unit", "")
        if not correct_answer:
            correct_answer = prob_data.get("correct_answer", "")
        if not grading_rules:
            grading_rules = prob_data.get("grading_rules", "")
        if not problem_text:
            problem_text = prob_data.get("statement", "")
        model_answer = prob_data.get("model_answer", "")
        required_reasoning = prob_data.get("required_reasoning", "")

        if not problem_image or not solution_image:
            return jsonify({"success": False, "error": "problem_image_b64와 solution_image_b64 필수"}), 400

        print(f"\n[채점 요청] 문제 이미지={len(problem_image)//1024}KB, "
              f"풀이 이미지={len(solution_image)//1024}KB, 참고 단계={len(solution_steps)}개")

        # 참고 텍스트 조립
        steps_str = "\n".join([f"  Step {s.get('index')}: {s.get('text')}" for s in solution_steps])
        if not steps_str:
            steps_str = "(없음)"

        # 루브릭 정보 조립
        rubric_list = prob_data.get("rubric", [])
        if rubric_list:
            rubric_str = "\n".join([
                f"  {i+1}. {r.get('criterion', '')} ({r.get('max_points', 25)}점): {r.get('description', '')}"
                for i, r in enumerate(rubric_list)
            ])
            total_max = sum(r.get("max_points", 0) for r in rubric_list)
        else:
            rubric_str = "  1. 개념 사용 (25점)\n  2. 근거 제시 (25점)\n  3. 계산·표현 (25점)\n  4. 결론 (25점)"
            total_max = 100

        prompt = f"""당신은 고등학교 수학 채점 전문가입니다.

두 이미지를 보고 학생 풀이를 채점하세요:
- 첫 번째 이미지: 원본 문제 (또는 풀이와 동일)
- 두 번째 이미지: 학생의 손글씨 풀이

[문제 텍스트]
{problem_text if problem_text else "(이미지에서 직접 확인)"}

[모범답안]
{model_answer if model_answer else "(없음)"}

[필수 풀이 근거]
{required_reasoning if required_reasoning else "(없음)"}

[알려진 정답]
{correct_answer if correct_answer else "(없음, 스스로 풀어 확인)"}

[학생 풀이 OCR (참고, 이미지가 항상 우선)]
{steps_str}

[채점 원칙]
1. 문제를 직접 풀어 정답을 스스로 구한 뒤 학생 풀이를 채점하세요.
2. 학생이 다른 방법을 써도 논리적으로 타당하면 ok.
3. 표기 실수는 관대하게, 실제 수학적 오류만 wrong으로.
4. 최초 오류 단계만 wrong, 그 이후는 depends.
5. 최종 답이 맞고 논리에 큰 결함 없으면 is_correct=true.

[부분점수 루브릭 - 반드시 각 기준별로 점수를 매기세요]
총 배점: {total_max}점
{rubric_str}

각 루브릭 기준에 대해:
- points: 0 ~ max_points 사이 정수. 기준을 완벽히 충족하면 만점, 전혀 못하면 0점.
- reason: 왜 이 점수를 줬는지 한 문장 (감점 사유 또는 만점 근거).
- evidence: 학생 풀이에서 해당 부분 직접 인용 (없으면 빈 문자열).
total_score: 모든 루브릭 points의 합산.

[특성 벡터 작성 지침]
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

        response = get_claude().messages.create(
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
        classification = classify_error(features, unit=unit)

        # 루브릭 점수 추출
        rubric_scores = grading.get("rubric_scores", [])
        total_score = grading.get("total_score", 0)
        if not total_score and rubric_scores:
            total_score = sum(r.get("points", 0) for r in rubric_scores)

        print(f"  [루브릭] {len(rubric_scores)}개 기준, 총점={total_score}")
        for rs in rubric_scores:
            print(f"    {rs.get('criterion')}: {rs.get('points')}/{rs.get('max_points')} - {rs.get('reason','')[:40]}")

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
            "steps": grading.get("steps", []),
            "rubric_scores": rubric_scores,
            "total_score": total_score,
            "features": features,
            "observed_errors": grading.get("observed_errors", []),
            "classification": classification,
            "feedback": feedback,
            "timestamp": now_iso(),
        }

        print(f"  [채점 완료] correct={result['is_correct']}, "
              f"primary={classification['primary_h']} ({classification['primary_label']}), "
              f"total={total_score}, gap={classification['gap']}")
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

        response = get_claude().messages.create(
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

        response = get_claude().messages.create(
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
        os.path.join(os.getcwd(), "manifest.json"),
        "/app/manifest.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "application/json"}


# ═══════════════════════════════════════════════════════
# 엔드포인트 7: 일관성 측정 (연구용)
# ═══════════════════════════════════════════════════════
@app.route("/measure_consistency", methods=["POST"])
def measure_consistency_endpoint():
    """
    같은 입력으로 채점을 여러 번 실행해서 재현 일관성을 측정.
    연구 목적 (Cohen's κ, 벡터 안정성 등).

    입력:
      problem_image_b64, solution_image_b64
      n_runs: 반복 횟수 (기본 3, 최대 5)
      grading_rules (선택)
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

        # 통계 계산
        primaries = [r["classification"]["primary_h"] for r in runs if r.get("success")]
        from collections import Counter
        counter = Counter(primaries)
        most_common, count = counter.most_common(1)[0] if counter else ("N/A", 0)

        # 벡터 표준편차
        vector_stability = {}
        for k in FEATURE_KEYS:
            values = [r["features"].get(k, 0.5) for r in runs if r.get("success")]
            if len(values) >= 2:
                vector_stability[k] = round(statistics.stdev(values), 4)
            else:
                vector_stability[k] = 0.0

        return jsonify({
            "success": True,
            "n_runs": n_runs,
            "code_consistency": round(count / n_runs, 4) if n_runs else 0,
            "primary_distribution": dict(counter),
            "vector_stability": vector_stability,
            "runs": runs,
            "timestamp": now_iso(),
        })

    except Exception as e:
        print(f"  [일관성 측정 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════
# 관리 API: 문제 & 채점 가이드라인 CRUD
# ═══════════════════════════════════════════════════════

# 인메모리 저장소 (서버 재시작하면 초기화. Supabase가 영구 저장)
_problems_db = {}   # id -> problem dict
_results_db = {}    # id -> grading result dict
_next_id = 1


# ─── Supabase 저장 헬퍼 ───
def save_to_supabase(result_data: dict):
    """채점 결과를 Supabase에 저장 (실패해도 서버는 계속 동작)"""
    sb = get_supabase()
    if not sb:
        print("  [Supabase] 클라이언트 없음, 스킵")
        return

    try:
        cls = result_data.get("classification", {})
        fb = result_data.get("feedback", {})
        problem_id = result_data.get("problem_id", "")

        # 1. submissions 저장
        submission_id = None
        try:
            sub_row = {
                "problem_id": problem_id,
                "ocr_text": result_data.get("ocr_text", ""),
                "ocr_steps": result_data.get("solution_steps"),
                "attempt_number": 1,
            }
            sub_resp = sb.table("submissions").insert(sub_row).execute()
            if sub_resp.data:
                submission_id = sub_resp.data[0].get("submission_id")
                print(f"  [Supabase] submission 저장 완료 (id={submission_id})")
        except Exception as e:
            print(f"  [Supabase] submission 저장 실패 (무시): {e}")

        # 2. gradings 저장
        grading_id = None
        grade_row = {
            "problem_id": problem_id,
            "is_correct": result_data.get("is_correct", False),
            "student_answer": result_data.get("student_final_answer", ""),
            "graded_steps": result_data.get("steps"),
            "observed_errors": result_data.get("observed_errors"),
            "features": result_data.get("features"),
            "primary_h": cls.get("primary_h", ""),
            "primary_distance": cls.get("primary_distance"),
            "secondary_h": cls.get("secondary_h"),
            "secondary_distance": cls.get("secondary_distance"),
            "gap": cls.get("gap"),
            "distribution": cls.get("distribution"),
            "feedback_text": fb.get("feedback", ""),
            "next_focus": fb.get("next_focus"),
            "total_score": result_data.get("total_score"),
            "model_version": "v2",
        }
        if submission_id:
            grade_row["submission_id"] = submission_id

        grade_resp = sb.table("gradings").insert(grade_row).execute()
        if grade_resp.data:
            grading_id = grade_resp.data[0].get("grading_id")
            print(f"  [Supabase] grading 저장 완료 (id={grading_id})")

        # 3. rubric_scores 저장 (문제의 루브릭 + AI 채점 결과가 있으면)
        if grading_id and problem_id:
            try:
                # DB에서 이 문제의 루브릭 조회
                rub_resp = sb.table("rubrics").select("*").eq("problem_id", problem_id).order("sort_order").execute()
                rubrics = rub_resp.data if rub_resp.data else []

                # 인메모리에서 루브릭 점수 (AI가 반환했으면)
                rubric_scores_data = result_data.get("rubric_scores", [])

                if rubrics and rubric_scores_data:
                    for i, rub in enumerate(rubrics):
                        score_data = rubric_scores_data[i] if i < len(rubric_scores_data) else {}
                        row = {
                            "grading_id": grading_id,
                            "rubric_id": rub["rubric_id"],
                            "points": score_data.get("points", 0),
                            "max_points": rub["max_points"],
                            "reason": score_data.get("reason", ""),
                            "evidence": score_data.get("evidence", ""),
                        }
                        sb.table("rubric_scores").insert(row).execute()
                    print(f"  [Supabase] rubric_scores {len(rubrics)}개 저장 완료")
            except Exception as e:
                print(f"  [Supabase] rubric_scores 저장 실패 (무시): {e}")

    except Exception as e:
        print(f"  [Supabase] 채점 저장 실패: {e}")


def load_problems_from_supabase():
    """Supabase에서 문제 목록 로드 → 인메모리에 병합"""
    sb = get_supabase()
    if not sb:
        return

    try:
        resp = sb.table("problems").select("*").eq("is_active", True).execute()
        if resp.data:
            for p in resp.data:
                pid = p.get("problem_id")
                if pid and pid not in _problems_db:
                    # Supabase 필드명 → 서버 내부 필드명 매핑
                    _problems_db[pid] = {
                        "id": pid,
                        "unit": p.get("unit_id", ""),
                        "type": p.get("type", "서술형"),
                        "title": p.get("title", ""),
                        "statement": p.get("statement", ""),
                        "choices": p.get("choices"),
                        "correct_answer": p.get("correct_answer", ""),
                        "score": p.get("score", 100),
                        "model_answer": p.get("model_answer", ""),
                        "required_reasoning": p.get("required_reasoning", ""),
                        "grading_rules": p.get("grading_rules", ""),
                        "expected_errors": p.get("expected_errors", []),
                        "feedback_example": p.get("feedback_example", ""),
                        "concept_tags": p.get("concept_tags", []),
                        "chapter": "함수와 그래프",
                    }
            print(f"  [Supabase] {len(resp.data)}개 문제 로드")
    except Exception as e:
        print(f"  [Supabase] 문제 로드 실패 (인메모리 사용): {e}")


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
        "difficulty": data.get("difficulty", "중"),          # 하/중/상/중상
        "title": data.get("title", ""),
        "statement": data.get("statement", ""),              # 문제 지문
        "choices": data.get("choices"),                      # 객관식 보기 (없으면 null)
        "answer_index": data.get("answer_index"),            # 객관식 정답 인덱스
        "correct_answer": data.get("correct_answer", ""),    # 서술형 정답
        "problem_image_url": data.get("problem_image_url", ""),

        # 채점 가이드라인 (교수님 문서 기반)
        "grading_rules": data.get("grading_rules", ""),             # 전체 채점 룰 텍스트
        "required_reasoning": data.get("required_reasoning", ""),   # 필수 풀이 근거
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


# ═══════════════════════════════════════════════════════
# 학생 결과 저장/조회 API
# ═══════════════════════════════════════════════════════
@app.route("/results", methods=["POST"])
def save_result():
    """채점 결과 저장"""
    data = request.get_json()
    rid = f"R{len(_results_db) + 1:06d}"

    result = {
        "result_id": rid,
        "problem_id": data.get("problem_id"),
        "student_id": data.get("student_id", "anonymous"),
        "is_correct": data.get("is_correct", False),
        "student_final_answer": data.get("student_final_answer", ""),
        "error_step_index": data.get("error_step_index"),
        "features": data.get("features", {}),
        "classification": data.get("classification", {}),
        "feedback": data.get("feedback", {}),
        "observed_errors": data.get("observed_errors", []),
        "steps": data.get("steps", []),
        "ocr_text": data.get("ocr_text", ""),
        "solution_image_b64": data.get("solution_image_b64", "")[:100] + "...(truncated)",  # 미리보기만
        "self_check": data.get("self_check", ""),   # 알고 풀었다/헷갈렸다/찍었다
        "created_at": now_iso(),
    }
    _results_db[rid] = result
    # Supabase에도 저장
    save_to_supabase(result)
    print(f"  [결과 저장] {rid}: problem={result['problem_id']}, correct={result['is_correct']}, H={result['classification'].get('primary_h', '?')}")
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
    codes = [r["classification"].get("primary_h", "?")
             for r in _results_db.values()
             if not r.get("is_correct", True)]
    counter = Counter(codes)
    return jsonify({
        "success": True,
        "total_wrong": sum(counter.values()),
        "distribution": dict(counter),
    })


# ═══════════════════════════════════════════════════════
# 교수님 문서 기반 문제 초기 등록 (서버 시작 시)
# ═══════════════════════════════════════════════════════
def load_initial_problems():
    """교수님 문서의 문항 1~10 초기 등록"""
    initial = [
        {
            "id": "Q01", "unit": "함수", "type": "서술형", "difficulty": "중",
            "title": "대응 관계가 함수인지 판정하기",
            "statement": "집합 X={1,2,3}, Y={a,b,c}에 대하여 두 대응 A, B가 있다.\nA: 1→a, 2→b, 3→b\nB: 1→a, 2→b와 2→c, 3→a\nA와 B가 각각 X에서 Y로의 함수인지 판정하고, 그 이유를 쓰시오.",
            "correct_answer": "A는 함수, B는 함수가 아님",
            "required_reasoning": "각 x가 Y의 원소 하나에만 대응하는지 확인한다.",
            "expected_errors": ["H1", "H10"],
            "model_answer": "A는 함수이다. X의 각 원소 1, 2, 3이 각각 Y의 원소 하나에만 대응한다. 2와 3이 같은 b에 대응해도 함수 조건에는 어긋나지 않는다. B는 함수가 아니다. 입력값 2가 b와 c 두 원소에 동시에 대응하므로 함수가 아니다.",
            "feedback_example": "좋은 풀이는 A와 B를 각각 확인합니다. 특히 B에서는 입력값 2 하나만 자세히 보면 함수가 아닌 이유가 드러납니다.",
            "rubric": [
                {"criterion": "개념 사용", "description": "함수의 조건을 '각 입력에 출력이 오직 하나'로 제시한다."},
                {"criterion": "근거 제시", "description": "A와 B에서 각 입력의 대응 여부를 각각 확인한다."},
                {"criterion": "계산·표현", "description": "대응 기호 또는 문장으로 A, B의 차이를 명확히 표현한다."},
                {"criterion": "결론", "description": "A는 함수, B는 함수 아님을 이유와 함께 쓴다."},
            ],
            "grading_rules": "함수 판정의 기본 조건(각 입력에 출력 오직 하나)을 확인했는지가 핵심이다.",
            "concept_tags": ["함수", "대응"],
            "score": 100,
        },
        {
            "id": "Q02", "unit": "함수", "type": "서술형", "difficulty": "중",
            "title": "그래프가 함수인지 판정하기",
            "statement": "좌표평면 위의 두 그래프 G1(y=x²-1)과 G2(x=y²-1)가 y를 x의 함수로 나타내는 그래프인지 판정하고, 판정 기준을 설명하시오.",
            "correct_answer": "G1은 함수의 그래프, G2는 함수의 그래프가 아님",
            "required_reasoning": "같은 x값에서 서로 다른 y값이 나오는지 확인한다.",
            "expected_errors": ["H3", "H10"],
            "model_answer": "G1은 y=x²-1이므로 임의의 x값에 대하여 y값이 하나로 정해져 함수의 그래프이다. G2는 x=y²-1이므로 예를 들어 x=0일 때 y=1 또는 y=-1이 가능하다. 같은 x값에 두 y값이 대응되므로 함수의 그래프가 아니다.",
            "rubric": [
                {"criterion": "개념 사용", "description": "그래프 판정 기준을 같은 x값에 대한 y값의 유일성으로 설명한다."},
                {"criterion": "근거 제시", "description": "G2에서 x=0일 때 y가 두 값임을 예로 든다."},
                {"criterion": "결론", "description": "G1은 함수의 그래프, G2는 아님을 명확히 쓴다."},
            ],
            "grading_rules": "수직선 판정 또는 구체적 x값 대입으로 판정했는지가 핵심이다.",
            "concept_tags": ["함수의 그래프", "수직선 판정"],
            "score": 100,
        },
        {
            "id": "Q03", "unit": "함수", "type": "서술형", "difficulty": "중",
            "title": "정의역, 공역, 치역 구하기",
            "statement": "함수 f:{-2,-1,0,1,2}→R가 f(x)=x²-1로 정의되어 있다. 이 함수의 정의역, 공역, 치역을 각각 구하고, 치역을 구한 과정을 쓰시오.",
            "correct_answer": "정의역 {-2,-1,0,1,2}, 공역 R, 치역 {-1,0,3}",
            "required_reasoning": "정의역 원소를 모두 대입하여 실제 함숫값의 집합을 만든다.",
            "expected_errors": ["H2", "H10"],
            "concept_tags": ["정의역", "공역", "치역"],
            "score": 100,
        },
        {
            "id": "Q04", "unit": "함수", "type": "서술형", "difficulty": "중",
            "title": "일대일함수와 일대일대응 판단하기",
            "statement": "함수 f:X→{a,b,c,d}에서 X={1,2,3}이고 f(1)=a, f(2)=b, f(3)=c이다. 이 함수가 일대일함수인지, 일대일대응인지 각각 판단하고 이유를 쓰시오.",
            "correct_answer": "일대일함수이지만 일대일대응은 아님",
            "required_reasoning": "서로 다른 입력이 서로 다른 출력으로 가는지, 치역과 공역이 같은지 확인한다.",
            "expected_errors": ["H4", "H10"],
            "concept_tags": ["일대일함수", "일대일대응"],
            "score": 100,
        },
        {
            "id": "Q05", "unit": "합성함수", "type": "서술형", "difficulty": "중",
            "title": "합성함수의 순서 설명하기",
            "statement": "두 함수 f(x)=2x-1, g(x)=x²+3에 대하여 (f∘g)(-2)와 (g∘f)(-2)를 각각 구하시오. 두 값이 달라지는 이유를 합성 순서와 관련하여 설명하시오.",
            "correct_answer": "(f∘g)(-2)=13, (g∘f)(-2)=28",
            "required_reasoning": "안쪽 함수가 먼저 적용되고 바깥 함수가 나중에 적용됨을 계산 과정에 드러낸다.",
            "expected_errors": ["H5", "H10"],
            "concept_tags": ["합성함수", "대응 순서"],
            "score": 100,
        },
        {
            "id": "Q06", "unit": "합성함수", "type": "서술형", "difficulty": "중",
            "title": "합성이 가능한 조건 판단하기",
            "statement": "함수 f:{1,2,3}→{4,5}, g:{4,5}→{0,1}. g∘f가 정의되는지 판단하고, (g∘f)(2)를 구하시오. 또한 f∘g가 정의되기 어려운 이유를 설명하시오.",
            "correct_answer": "g∘f는 정의됨, (g∘f)(2)=1, f∘g는 정의 어려움",
            "required_reasoning": "f의 출력이 g의 입력으로 들어갈 수 있는지 확인한다.",
            "expected_errors": ["H5", "H2"],
            "concept_tags": ["합성함수", "정의역", "공역"],
            "score": 100,
        },
        {
            "id": "Q07", "unit": "역함수", "type": "서술형", "difficulty": "중상",
            "title": "역함수 존재 여부 판단하기",
            "statement": "두 함수 f:{1,2,3}→{2,4,6}, f(x)=2x와 g:{-1,0,1}→{0,1}, g(x)=x²가 있다. 각 함수의 역함수가 존재하는지 판단하고, 존재한다면 한 가지 역함숫값을 구하시오.",
            "correct_answer": "f는 역함수 존재 (f⁻¹(6)=3), g는 존재하지 않음",
            "required_reasoning": "각 함수가 일대일대응인지 확인한다.",
            "expected_errors": ["H6", "H4"],
            "concept_tags": ["역함수", "일대일대응"],
            "score": 100,
        },
        {
            "id": "Q08", "unit": "역함수", "type": "서술형", "difficulty": "중상",
            "title": "역함수 식과 그래프 관계 설명하기",
            "statement": "함수 f(x)=3x-2의 역함수를 구하고, 함수 y=f(x)의 그래프와 역함수의 그래프가 어떤 관계인지 설명하시오.",
            "correct_answer": "f⁻¹(x)=(x+2)/3, y=x에 대하여 대칭",
            "required_reasoning": "y=3x-2에서 x를 y에 대한 식으로 나타낸 뒤 x와 y를 바꾼다.",
            "expected_errors": ["H7", "H8"],
            "concept_tags": ["역함수", "그래프 대칭"],
            "score": 100,
        },
        {
            "id": "Q09", "unit": "유리함수", "type": "서술형", "difficulty": "중상",
            "title": "유리함수의 정의역, 치역, 점근선 찾기",
            "statement": "함수 h(x)=2/(x-1)+3의 정의역, 치역, 점근선을 구하고, 이 그래프가 y=2/x의 그래프에서 어떻게 이동한 것인지 설명하시오.",
            "correct_answer": "정의역 x≠1, 치역 y≠3, 점근선 x=1, y=3, x축 방향 +1, y축 방향 +3",
            "required_reasoning": "분모가 0이 되는 x값과 y가 될 수 없는 값을 확인한다.",
            "expected_errors": ["H9", "H2", "H10"],
            "concept_tags": ["유리함수", "점근선", "그래프 변환"],
            "score": 100,
        },
        {
            "id": "Q10", "unit": "역함수", "type": "서술형", "difficulty": "상",
            "title": "잘못된 풀이의 오류 설명 및 수정하기",
            "statement": "다음 학생 풀이를 읽고 오류를 설명한 뒤, 올바르게 수정하시오.\n\n학생 풀이: 'y=x²의 역함수는 x=y²에서 x와 y를 바꾸면 y=x²이므로 다시 y=x²이다. 또는 양변에 제곱근을 씌워 y=√x라고 해도 된다. 따라서 y=x²의 역함수는 항상 존재한다.'",
            "correct_answer": "정의역이 실수 전체인 y=x²은 일대일함수가 아니므로 역함수가 존재하지 않음. 정의역을 x≥0으로 제한해야 역함수 y=√x가 성립.",
            "required_reasoning": "y=x²의 정의역이 실수 전체이면 일대일함수가 아님을 설명한다.",
            "expected_errors": ["H6", "H7", "H10"],
            "concept_tags": ["역함수", "정의역 제한", "일대일함수"],
            "score": 100,
        },
    ]

    for p in initial:
        p["chapter"] = "함수와 그래프"
        p["created_at"] = now_iso()
        if "rubric" not in p:
            p["rubric"] = []
        if "model_answer" not in p:
            p["model_answer"] = ""
        if "feedback_example" not in p:
            p["feedback_example"] = ""
        if "grading_rules" not in p:
            p["grading_rules"] = ""
        _problems_db[p["id"]] = p

    print(f"  [초기 문제 등록] {len(initial)}개 문제 로드 완료")


# ─── 초기 문제 로드 (모듈 로드 시 실행, gunicorn 호환) ───
load_initial_problems()
load_problems_from_supabase()  # DB에 있는 문제도 병합


# ─── 서버 시작 (로컬 실행용) ──────────────────────────
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("=" * 60)
    print("  hongsu v2 서버 (vision + tool_use + 벡터 분류)")
    print(f"  로컬: http://localhost:5000")
    print(f"  네트워크: http://{local_ip}:5000")
    print(f"  학생 앱: http://{local_ip}:5000/student")
    print(f"  관리자:  http://{local_ip}:5000/admin")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
