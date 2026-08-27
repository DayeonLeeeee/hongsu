"""
파이프라인:
  1. 풀이 OCR:     Mathpix + Gemini Flash 병렬 → 단계별 평문 (표시용)
  2. 자연어 수정:  Claude Haiku (vision + tool_use)
  3. 채점:         Claude Opus 4.6 (vision + tool_use)
                    → 13차원 특성 벡터 + observed_errors
                    → 코드로 유클리드 거리 → 최근접 H코드
  4. 개인화 피드백: Claude Haiku (tool_use)
  5. 모범답안

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

# 모델 선택은 실데이터 비교 후 모델 변경 또는 파인튜닝 가능성 검토 예정.
MODEL_OPUS   = "claude-opus-4-6"
MODEL_HAIKU  = "claude-haiku-4-5-20251001"
MODEL_GEMINI = "gemini-2.5-flash"
PROMPT_VERSION = "v2.1-2026-08-28"  # 변경 시 날짜와 함께 갱신

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


# ─── 2. Gemini Flash OCR (평문) ─────
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


# 오답 유형 프로파일 (H1~H10, 문서)

# 현재 수치는 오세준 교수님 문서 기반 수작업 추정치.
# 합성 데이터(가상 학생 오답) 생성 → 반복 실험으로 프로파일 값을 지속 조정 필요
# H코드 간 프로파일이 가까우면 분류 경계가 불분명해지므로 각 H코드가 충분히 구별되는 수치인지 합성 데이터로 확인 필요.
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

RECOMMENDED_FLOW = {
    "H1": "대응표 판정 → 화살표 그림 판정 → 비함수 예시 고치기",
    "H2": "정의역 표시 → 함숫값 계산 → 치역 집합 쓰기",
    "H3": "점 집합 판정 → 그래프 판정 → 판정 이유 문장 쓰기",
    "H4": "중복 출력 찾기 → 치역과 공역 비교 → 역함수 가능성 판단",
    "H5": "입력 → 안쪽 함수 → 바깥 함수 흐름도 쓰기",
    "H6": "일대일 판정 → 정의역 제한 → 역함수 구하기",
    "H7": "y=f(x) → x=… → x,y 교환 → 정의역 확인",
    "H8": "점 (a,b) → (b,a) 변환 → 그래프 대칭 설명",
    "H9": "제한 조건 찾기 → 기준 그래프 → 평행이동 → 점근선/치역",
    "H10": "판정 기준 문장 → 풀이 근거 표시 → 결론 재서술",
}


# 벡터 분류 유틸

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
    """현재 계산 방식: (추후 변경 가능 높음)
    13차원 벡터 → 각 H와의 거리 + 최근접 유형 + 분포
    unit이 주어지면 해당 단원과 무관한 H코드에 페널티를 줘서 분포를 명확히 함.

    [거리 방식]
    현재 유클리드 거리 + 단원 페널티 방식.
    축 가중치(현재 균등 1:1), 거리 함수, 페널티 계수 등은 실데이터 기반 최적화 예정.
    대안(코사인 유사도, 가중 유클리드, 클러스터링 등)도 비교 검토 중.

    [근소 차이 판정]
    1위와 2위의 거리 차(gap)가 매우 작을 때(예: 0.459 vs 0.458) 순수 수치만으로
    최종 H코드를 결정하는 것이 적절한지 논의 필요.
    고려 중인 방안:
      - gap < 임계값이면 "복합 유형"으로 보고 1위·2위 동시 리포트
      - 교사 확인(teacher_labels)을 통해 근소 차이 케이스의 판정 기준 수립
      - 단원·문제 맥락을 가중치로 반영하여 경계 케이스 해소

    [교사-AI 판정 일치도]
    현직 교사가 판단하는 오답 원인과 LLM이 추출한 특성 벡터 기반 분류가 일치하지 않을 수 있음. 이 오차(Cohen's κ로 측정 예정)를 어떻게 반영할지 논의 필요.
      - 교사 라벨과 AI 분류가 불일치하는 패턴을 수집하여 프로파일 보정
      - 특정 H코드에서 체계적 불일치가 발견되면 해당 프로파일 재설계
      - 불일치율이 높으면 특성 벡터 설계 자체를 재검토
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
# H코드 직접 분류기 (LLM 기반 — 벡터 방식과 병행)
# ═══════════════════════════════════════════════════════
H_CRITERIA = {
    "H1": {
        "name": "함수 정의 오류",
        "definition": "입력값 하나에 출력값이 오직 하나 대응해야 한다는 함수의 정의를 확인하지 않음.",
        "indicators": "유일성 조건 미확인, '모든 y가 쓰여야 한다'고 오해, 서로 다른 x가 같은 y로 가면 안 된다고 오해.",
    },
    "H2": {
        "name": "정의역·공역·치역 혼동",
        "definition": "정의역, 공역, 치역 세 개념을 구분하지 못하거나 혼동.",
        "indicators": "공역을 치역으로 그대로 사용, 함숫값 계산 없이 치역 결정, 정의역 원소를 빠뜨리고 치역 작성.",
    },
    "H3": {
        "name": "그래프 판정 오류",
        "definition": "그래프가 함수인지 판정할 때 같은 x값에 y값이 하나인지를 확인하지 않음.",
        "indicators": "수직선 판정 미사용, '곡선이 하나라서 함수'라고 판단, 구체적 x값 대입 없이 판정.",
    },
    "H4": {
        "name": "일대일함수·일대일대응 오류",
        "definition": "일대일함수와 일대일대응의 차이를 혼동하거나, 판정 조건을 빠뜨림.",
        "indicators": "서로 다른 입력→서로 다른 출력 확인 누락, 치역=공역 조건 미확인, 두 개념을 같은 것으로 취급.",
    },
    "H5": {
        "name": "합성함수 순서 오류",
        "definition": "합성함수에서 안쪽 함수를 먼저 적용해야 하는 순서를 틀림.",
        "indicators": "f∘g에서 f를 먼저 적용, (f∘g)(x)=f(x)·g(x)로 오해, 합성 순서 설명 없이 계산만.",
    },
    "H6": {
        "name": "역함수 존재 조건 오류",
        "definition": "역함수가 존재하려면 원래 함수가 일대일대응이어야 한다는 조건을 확인하지 않음.",
        "indicators": "일대일대응 확인 없이 역함수 구함, 정의역 제한 필요성 무시, 식 변환만으로 역함수 존재 판단.",
    },
    "H7": {
        "name": "역함수 식 변형 오류",
        "definition": "y=f(x)에서 x를 y로 나타낸 뒤 x와 y를 바꾸는 과정에서 실수.",
        "indicators": "x,y 교환 누락, 두 단계를 한 번에 처리하다 실수, 역함수의 정의역·치역 미확인.",
    },
    "H8": {
        "name": "역함수 그래프 오류",
        "definition": "역함수 그래프가 y=x 대칭이라는 관계를 잘못 이해하거나 적용하지 못함.",
        "indicators": "(a,b)→(b,a) 변환 미적용, y=x 대칭이 아닌 다른 대칭 주장, 그래프 대칭 설명 생략.",
    },
    "H9": {
        "name": "유리·무리함수 확장 오류",
        "definition": "유리함수·무리함수에서 정의역 제한(분모≠0, 근호≥0)을 확인하지 않음.",
        "indicators": "분모=0 조건 무시, 점근선 미확인, 그래프 변환(평행이동) 전 정의역 제한 생략.",
    },
    "H10": {
        "name": "근거·정당화 부족",
        "definition": "답은 맞거나 근접하지만, 왜 그런지 근거나 풀이 과정을 서술하지 않음.",
        "indicators": "답만 기술하고 과정 없음, 사용한 조건·정의 명시 없음, 판정 기준 생략.",
    },
}

TOOL_CLASSIFY_H = {
    "name": "classify_h_code",
    "description": "학생 풀이에서 관찰된 오류를 H1~H10 판정기준에 따라 직접 분류합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "errors": {
                "type": "array",
                "description": "wrong으로 판정된 각 단계의 오류 분류. H10은 풀이 전체 판정.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_index": {"type": "integer", "description": "오류 단계 번호. H10(근거 부족)이면 0."},
                        "h_code": {"type": "string", "description": "H1~H10 중 하나"},
                        "reason": {"type": "string", "description": "이 유형으로 판정한 이유 (한 문장)"},
                        "evidence": {"type": "string", "description": "학생 풀이에서 해당 오류의 근거가 되는 원문 인용"},
                    },
                    "required": ["step_index", "h_code", "reason", "evidence"],
                },
            },
            "primary_h": {"type": "string", "description": "주 오류 유형 (첫 wrong 단계의 H코드, 또는 가장 핵심적인 오류)"},
            "secondary_h": {
                "type": ["string", "null"],
                "description": "보조 오류 유형 (다른 wrong 단계의 H코드). 없으면 null.",
            },
            "no_error": {"type": "boolean", "description": "오류가 전혀 없으면 true"},
        },
        "required": ["errors", "primary_h", "no_error"],
    },
}


def classify_error_direct(grading_result: dict, problem_text: str, solution_text_or_steps: str) -> dict:
    """LLM이 H1~H10 판정기준을 직접 보고 분류. 벡터 방식과 병행."""
    try:
        steps = grading_result.get("steps", [])
        wrong_steps = [s for s in steps if s.get("status") == "wrong"]

        if not wrong_steps:
            return {"method": "direct", "no_error": True, "errors": [], "primary_h": None, "secondary_h": None}

        # H코드 판정기준 조립
        criteria_text = "\n".join([
            f"  {code}: {info['name']}\n    정의: {info['definition']}\n    지표: {info['indicators']}"
            for code, info in H_CRITERIA.items()
        ])

        wrong_desc = "\n".join([
            f"  Step {s['index']}: [{s['status']}] {s.get('text','')} — {s.get('explanation','')}"
            for s in steps
        ])

        prompt = f"""아래 학생 풀이의 채점 결과를 보고, wrong으로 판정된 각 단계의 오류를 H1~H10 중 하나로 분류하세요.

[문제]
{problem_text}

[학생 풀이 + 채점 결과]
{wrong_desc}

[관찰된 실수]
{', '.join(grading_result.get('observed_errors', []))}

[H코드 판정 기준]
{criteria_text}

[규칙]
1. wrong 단계마다 가장 잘 맞는 H코드 1개를 배정하세요.
2. 학생 풀이 원문에서 근거를 직접 인용(evidence)하세요.
3. 풀이 전체에 근거가 부족하면 step_index=0으로 H10을 추가할 수 있습니다.
4. primary_h = 첫 wrong 단계의 H코드. secondary_h = 다른 wrong 단계의 H코드 (있으면).
5. 오류가 없으면 no_error=true.

반드시 classify_h_code tool로만 응답하세요."""

        response = get_claude().messages.create(
            model=MODEL_OPUS,
            max_tokens=2048,
            temperature=0.0,
            tools=[TOOL_CLASSIFY_H],
            tool_choice={"type": "tool", "name": "classify_h_code"},
            messages=[{"role": "user", "content": prompt}],
        )
        result = parse_tool_use(response)
        result["method"] = "direct"
        return result
    except Exception as e:
        print(f"  [직접 분류 실패] {e}")
        return {"method": "direct", "no_error": False, "errors": [], "primary_h": None, "error_msg": str(e)}


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

# ─── (3) 채점 (특성 벡터 추출) ────
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

# ─── (4) 개인화 피드백 ────
TOOL_PERSONAL_FEEDBACK = {
    "name": "compose_feedback",
    "description": "학생 실수와 오답 유형을 종합해 짧은 피드백과 상세 피드백을 생성합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "학생에게 줄 핵심 피드백 (2문장 이내, 안내 어조)",
            },
            "detailed_feedback": {
                "type": "string",
                "description": "학생이 추가 설명을 원할 때 보여줄 상세 피드백 (3~5문장). 왜 틀렸는지, 올바른 접근은 무엇인지, 비슷한 문제에서 주의할 점까지 포함.",
            },
            "next_focus": {
                "type": "array",
                "items": {"type": "string"},
                "description": "다음에 학생이 확인/연습할 것 1~2개",
            },
        },
        "required": ["feedback", "detailed_feedback", "next_focus"],
    },
}


# 엔드포인트 1: OCR
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


# 엔드포인트 4: 채점 (특성 벡터 + 최근접 분류)
@app.route("/grade", methods=["POST"])
def grade_endpoint():
    """
    Claude Opus vision — 학생 풀이 이미지를 보고 채점.
    문제 정보는 DB에서 텍스트로 전체 조회.
    → 13차원 특성 벡터 뽑음
    → 코드로 최근접 H코드 결정
    → Claude Haiku로 개인화 피드백 생성

    입력:
      solution_image_b64: 학생 풀이 이미지 (이미지 또는 텍스트 중 하나 필수)
      solution_text: 학생 풀이 텍스트 (합성답안 실험용, 이미지 없을 때 사용)
      problem_id: DB 문제 ID (필수 — 문제 텍스트·정답·루브릭 등 조회)
      solution_steps: [{index, text}] OCR 평문 단계별 (참고용, 선택)
      unit: 단원명 (선택, problem_id로 자동 조회)
    """
    try:
        data = request.get_json()
        solution_image = data.get("solution_image_b64", "")
        solution_text = data.get("solution_text", "")
        problem_text = data.get("problem_text", "")
        solution_steps = data.get("solution_steps", [])
        correct_answer = data.get("correct_answer", "")
        grading_rules = data.get("grading_rules", "")
        problem_id = data.get("problem_id", "")
        unit = data.get("unit", "")
        grading_mode = "image" if solution_image else "text" if solution_text else None

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
        observation_points = prob_data.get("observation_points", "")

        if not grading_mode:
            return jsonify({"success": False, "error": "solution_image_b64 또는 solution_text 중 하나 필수"}), 400

        print(f"\n[채점 요청] mode={grading_mode}, "
              f"problem_id={problem_id}, 참고 단계={len(solution_steps)}개")

        # 참고 텍스트 조립
        steps_str = "\n".join([f"  Step {s.get('index')}: {s.get('text')}" for s in solution_steps])
        if not steps_str:
            steps_str = "(없음)"

        # 루브릭 정보 조립
        rubric_list = prob_data.get("rubric", [])
        if rubric_list:
            rubric_str = "\n".join([
                f"  {i+1}. {r.get('criterion', '')} 
