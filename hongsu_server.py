"""
파이프라인:
  1. 풀이 OCR:     Mathpix + Gemini Flash 병렬 → 단계별 평문 (표시용)
  2. 자연어 수정:  Claude Haiku (vision + tool_use)ss
  3. 채점:         Claude Opus 4.6 (vision + tool_use)
                    → observed_errors
                    → 직접분류기 사용. 
  4. 개인화 피드백: Claude Haiku (tool_use)
  5. 모범답안

[R v2 이식 — hongsu-rv2-1.0]
  /grade      R v2 진단 (이용오 선생님 키트 로직): 줄별 ok/wrong/depends + R코드 + J수준
              - 문항·코드북·줄별 기준판정·판정규칙은 전부 엑셀(materials/dataset_rv2.xlsx)에서 조립
              - 문항별 허용 코드만 노출, 인용 원문 대조·허용 밖 코드 거부·1회 재시도, 기권(review_needed)
              - 응답 원문·토큰·prompt_version·엑셀 해시를 모든 결과에 저장 (Supabase rv2_gradings)
              - 피드백은 다음 단계 (이번 버전은 진단만)
  /grade_h    기존 H코드 파이프라인 (Opus 채점·100점 루브릭 → H분류 → Haiku 피드백), 변경 없음
  문항        기존 Q01~Q10 하드코딩 제거 → 엑셀 01_PROBLEMS(INV-R-01~10)로 교체
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
PROMPT_VERSION = "v2.2-2026-08-29"  # 변경 시 날짜와 함께 갱신

# 프롬프트 분리 모듈
import prompts as PROMPTS
from prompts import ocr as p_ocr
from prompts import refine as p_refine
from prompts import grading as p_grading
from prompts import classify as p_classify
from prompts import feedback as p_feedback
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


def _norm_for_evidence(text: str) -> str:
    """evidence 대조용 정규화: normalize_plain + 모든 공백 제거 + 소문자화.
    부호(-, +), 위첨자(²), 특수기호는 유지. 학생/모델의 표기 차이만 흡수."""
    s = normalize_plain(text or "")
    s = re.sub(r'\s+', '', s)
    return s.lower()


def _evidence_in_text(evidence: str, ocr_text: str) -> bool:
    """evidence가 OCR 텍스트에 실제 존재하는지 substring 대조."""
    if not evidence or not evidence.strip():
        return True  # 빈 인용은 허용 (Opus에 "없으면 빈 문자열" 지시함)
    ev = _norm_for_evidence(evidence)
    base = _norm_for_evidence(ocr_text)
    if not ev or not base:
        return True  # 대조 불가는 통과 처리
    return ev in base


def _validate_grading_evidence(grading: dict, ocr_base: str) -> list:
    """steps[].evidence + rubric_scores[].evidence 전수 검증.
    반환: [{"kind": "step"|"rubric", "id": ..., "evidence": ...}, ...] (실패 목록)"""
    if not ocr_base or not ocr_base.strip():
        return []  # OCR 원문 없으면 검증 스킵
    failures = []
    for i, s in enumerate(grading.get("steps", []) or []):
        if not isinstance(s, dict):
            continue
        ev = s.get("evidence", "")
        if ev and not _evidence_in_text(ev, ocr_base):
            failures.append({
                "kind": "step",
                "id": s.get("step_index", i + 1),
                "evidence": ev,
            })
    for i, r in enumerate(grading.get("rubric_scores", []) or []):
        if not isinstance(r, dict):
            continue
        # 감점형: 각 감점 내역의 evidence를 검증
        for j, d in enumerate(r.get("deductions", []) or []):
            if not isinstance(d, dict):
                continue
            ev = d.get("evidence", "")
            if ev and not _evidence_in_text(ev, ocr_base):
                failures.append({
                    "kind": "rubric_deduction",
                    "id": f"{r.get('criterion', f'#{i+1}')}/L{d.get('line_ref',0)}",
                    "evidence": ev,
                })
    return failures


# 오답 유형 프로파일 (H1~H10, 문서)

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


# 현재로서의 단원별 관련 H코드
UNIT_H_RELEVANCE = {
    "함수":     ["H1", "H2", "H3", "H10"],
    "역함수":   ["H4", "H6", "H7", "H8", "H10"],
    "합성함수": ["H5", "H2", "H10"],
    "유리함수": ["H9", "H2", "H10"],
    "무리함수": ["H9", "H2", "H10"],
}


# H코드 직접 분류기 (LLM 기반 — 현재 메인 분류 방식)
# 향후 교사 라벨 수집 후 classify_logistic.py(로지스틱 회귀), classify_dina.py(DINA)로 대체/병행 예정.
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
            "primary_h": {"type": ["string", "null"], "description": "주 오류 유형 (첫 wrong 단계의 H코드). wrong이 없으면 null."},
            "secondary_h": {
                "type": ["string", "null"],
                "description": "보조 오류 유형 (다른 wrong 단계의 H코드). 없으면 null.",
            },
            "h10_global": {
                "type": "object",
                "description": "H10(근거·정당화 부족)은 라인이 아닌 풀이 전체로 판정.",
                "properties": {
                    "applies": {"type": "boolean", "description": "풀이 전체가 근거 부족에 해당하면 true"},
                    "reason": {"type": "string", "description": "H10 판정 이유 (한 문장). applies=false면 빈 문자열."},
                },
                "required": ["applies", "reason"],
            },
            "no_error": {"type": "boolean", "description": "오류가 전혀 없으면 true"},
        },
        "required": ["errors", "primary_h", "h10_global", "no_error"],
    },
}


def classify_error_direct(grading_result: dict, problem_text: str, solution_text_or_steps: str) -> dict:
    """LLM이 H1~H10 판정기준을 직접 보고 분류."""
    try:
        steps = grading_result.get("steps", [])
        wrong_steps = [s for s in steps if s.get("status") == "wrong"]

        if not wrong_steps:
            # wrong 없지만 H10은 별도 판정 대상 → 프롬프트에 넣어 판단시킴
            pass  # 이 경우도 아래 흐름으로 진입해서 h10_global만 확인

        # H코드 판정기준 조립
        criteria_text = "\n".join([
            f"  {code}: {info['name']}\n    정의: {info['definition']}\n    지표: {info['indicators']}"
            for code, info in H_CRITERIA.items()
        ])

        wrong_desc = "\n".join([
            f"  Step {s['index']}: [{s['status']}] {s.get('text','')} — {s.get('explanation','')}"
            for s in steps
        ])

        prompt = p_classify.build(
            problem_text=problem_text,
            wrong_desc=wrong_desc,
            observed_errors_str=', '.join(grading_result.get('observed_errors', [])),
            criteria_text=criteria_text,
        )

        response = get_claude().messages.create(
            model=MODEL_OPUS,
            max_tokens=2048,
            tools=[TOOL_CLASSIFY_H],
            tool_choice={"type": "tool", "name": "classify_h_code"},
            messages=[{"role": "user", "content": prompt}],
        )
        result = parse_tool_use(response)
        result["method"] = "direct"

        # Edge Case: 라인당 복수 오류 자동 감지 (reason에 "복합 오류 감지" 포함 여부)
        multi_error_lines = []
        for err in result.get("errors", []):
            if "복합 오류" in err.get("reason", ""):
                multi_error_lines.append({
                    "step_index": err.get("step_index"),
                    "selected_h": err.get("h_code"),
                    "reason": err.get("reason"),
                })
        result["multi_error_flag"] = len(multi_error_lines) > 0
        result["multi_error_lines"] = multi_error_lines
        if multi_error_lines:
            print(f"  ⚠️ 복수 오류 감지: {len(multi_error_lines)}개 라인 → 검수 큐 대상")

        return result
    except Exception as e:
        print(f"  [직접 분류 실패] {e}")
        return {
            "method": "direct", "no_error": False, "errors": [],
            "primary_h": None, "secondary_h": None,
            "h10_global": {"applies": False, "reason": ""},
            "multi_error_flag": False, "multi_error_lines": [],
            "error_msg": str(e),
        }


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

# ─── (2) 채점 ────
TOOL_GRADE_SOLUTION = {
    "name": "grade_student_solution",
    "description": "학생 풀이를 채점합니다: 루브릭별 부분점수 + 단계별 판정.",
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
                        "index": {"type": "integer", "description": "줄 번호 (1부터)"},
                        "text": {"type": "string", "description": "학생이 쓴 내용 (평문)"},
                        "status": {
                            "type": "string",
                            "enum": ["ok", "wrong", "depends", "insufficient", "unreadable", "not_attempted"],
                            "description": (
                                "ok=올바름, wrong=독립 오류, depends=앞 오류 전파, "
                                "insufficient=근거 부족(맞지만 서술 미흡), "
                                "unreadable=OCR/필기 판독 불가, "
                                "not_attempted=시도 자체가 없음"
                            ),
                        },
                        "explanation": {"type": "string", "description": "이 단계에 대한 짧은 설명"},
                        "depends_on_line_id": {
                            "type": ["integer", "null"],
                            "description": "status=depends일 때, 어느 줄에서 전파된 오류인지 (해당 줄 index). 다른 status면 null.",
                        },
                    },
                    "required": ["index", "text", "status", "explanation"],
                },
            },
            "rubric_scores": {
                "type": "array",
                "description": "감점형 루브릭 채점. 4기준 각 25점 만점, 감점 사유마다 근거 필수.",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {
                            "type": "string",
                            "enum": ["개념 사용", "근거 제시", "계산·표현", "결론"],
                            "description": "4기준 중 하나",
                        },
                        "max_points": {"type": "integer", "const": 25, "description": "각 기준 만점은 25점 고정"},
                        "deductions": {
                            "type": "array",
                            "description": "감점 내역. 감점 없으면 빈 배열. 냅다 점수 깎지 말고 근거마다 나열.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "points": {
                                        "type": "integer",
                                        "description": "감점 (음수. 예: -5). 25점을 한 번에 다 깎지 말고, 감점 척도표에 따라 나눠서 기록할 것.",
                                    },
                                    "reason": {"type": "string", "description": "감점 사유 (구체적으로)"},
                                    "line_ref": {
                                        "type": "integer",
                                        "description": "감점 근거가 있는 학생 풀이 줄 번호. 특정 줄 없으면 0.",
                                    },
                                    "evidence": {
                                        "type": "string",
                                        "description": "학생 풀이에서 감점 근거 원문 인용 (없으면 빈 문자열)",
                                    },
                                },
                                "required": ["points", "reason", "line_ref", "evidence"],
                            },
                        },
                        "score": {
                            "type": "integer",
                            "description": "이 기준의 최종 점수 = max(0, 25 - 감점 합계). 음수 불가.",
                            "minimum": 0,
                            "maximum": 25,
                        },
                    },
                    "required": ["criterion", "max_points", "deductions", "score"],
                },
            },
            "total_score": {
                "type": "integer",
                "description": "4기준 score 합계 (0~100). 100 - 전체 감점.",
                "minimum": 0,
                "maximum": 100,
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
            "observed_errors",
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


# 엔드포인트 3-H: 기존 H코드 채점 (변경 없음, 경로만 /grade → /grade_h)
@app.route("/grade_h", methods=["POST"])
def grade_endpoint():
    """
    Claude Opus vision — 학생 풀이 이미지를 보고 채점.
    문제 정보는 DB에서 텍스트로 전체 조회.
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
        attempt_count = int(data.get("attempt_count", 1))  # 프론트에서 재도전 횟수 전송
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

        prompt = p_grading.build(
            grading_mode=grading_mode,
            problem_text=problem_text,
            model_answer=model_answer,
            required_reasoning=required_reasoning,
            correct_answer=correct_answer,
            observation_points=observation_points,
            deduction_scale=prob_data.get("deduction_scale", {}),
            steps_str=steps_str,
            solution_text=solution_text,
        )

        # 모드별 Claude 호출
        if grading_mode == "image":
            user_content = [
                make_image_block(solution_image, "image/jpeg"),
                {"type": "text", "text": prompt},
            ]
        else:
            user_content = [{"type": "text", "text": prompt}]

        response = get_claude().messages.create(
            model=MODEL_OPUS,
            max_tokens=4096,
            tools=[TOOL_GRADE_SOLUTION],
            tool_choice={"type": "tool", "name": "grade_student_solution"},
            system="수학 채점 전문가. 학생 풀이를 정확히 채점합니다.",
            messages=[{"role": "user", "content": user_content}],
        )
        grading = parse_tool_use(response)

        # ─── 인용 검증 (evidence가 OCR/텍스트 원문에 실제 존재하는지) ───
        ocr_base = solution_text if grading_mode == "text" else "\n".join(
            [s.get("text", "") for s in solution_steps]
        )
        evidence_confidence = "high"
        evidence_failures = _validate_grading_evidence(grading, ocr_base)

        if evidence_failures:
            print(f"  [인용 검증] 실패 {len(evidence_failures)}건 → 재시도")
            for f in evidence_failures[:5]:
                print(f"    {f['kind']}({f['id']}): {f['evidence'][:60]}")

            # 재시도 프롬프트: 실패 인용 명시 + OCR 원문 강조
            failed_list = "\n".join([
                f"  - [{f['kind']} {f['id']}] \"{f['evidence']}\"" for f in evidence_failures
            ])
            retry_prompt = p_grading.build_retry(
                base_prompt=prompt,
                failed_list=failed_list,
                ocr_base=ocr_base,
            )
            if grading_mode == "image":
                user_content_retry = [
                    make_image_block(solution_image, "image/jpeg"),
                    {"type": "text", "text": retry_prompt},
                ]
            else:
                user_content_retry = [{"type": "text", "text": retry_prompt}]

            response2 = get_claude().messages.create(
                model=MODEL_OPUS,
                max_tokens=4096,
                tools=[TOOL_GRADE_SOLUTION],
                tool_choice={"type": "tool", "name": "grade_student_solution"},
                system="수학 채점 전문가. 학생 풀이를 정확히 채점합니다.",
                messages=[{"role": "user", "content": user_content_retry}],
            )
            grading = parse_tool_use(response2)
            evidence_failures = _validate_grading_evidence(grading, ocr_base)
            if evidence_failures:
                evidence_confidence = "low"
                print(f"  [인용 검증] 재시도도 실패 {len(evidence_failures)}건 → confidence=low")
            else:
                print(f"  [인용 검증] 재시도 통과")

        is_correct = grading.get("is_correct", False)
        has_wrong = any(s.get("status") == "wrong" for s in grading.get("steps", []))

        # ─── S4-5: 앞뒤 불일치 감지 ("정답 판정인데 오류 evidence 있음") ───
        contradiction_flags = []
        if is_correct:
            # 정답으로 판정했는데 wrong 단계가 있으면 모순
            wrong_steps_for_check = [s for s in grading.get("steps", []) if s.get("status") == "wrong"]
            if wrong_steps_for_check:
                contradiction_flags.append({
                    "type": "correct_but_wrong_steps",
                    "detail": f"is_correct=true인데 wrong 단계 {len(wrong_steps_for_check)}개 존재",
                    "step_indices": [s.get("index") for s in wrong_steps_for_check],
                })
            # 정답으로 판정했는데 감점이 크면 (total_score < 80) 모순
            interim_total = sum(rs.get("score", 0) for rs in grading.get("rubric_scores", []))
            if interim_total < 80:
                contradiction_flags.append({
                    "type": "correct_but_low_score",
                    "detail": f"is_correct=true인데 total_score={interim_total} (기준 80 미만)",
                })

        if contradiction_flags:
            print(f"  ⚠️ S4-5 모순 감지: {len(contradiction_flags)}건 → confidence=low, 검수 큐로")
            for cf in contradiction_flags:
                print(f"    [{cf['type']}] {cf['detail']}")
            evidence_confidence = "low"  # 모순도 confidence 강등 사유

        if is_correct and not has_wrong:
            # 정답 풀이 → wrong 없음. 그래도 H10(근거 부족)은 별도 판정.
            classification = classify_error_direct(grading, problem_text,
                                                   solution_text if grading_mode == "text" else steps_str)
            # 정답이면 primary/secondary는 강제로 없음. h10_global만 유효.
            classification["primary_h"] = None
            classification["secondary_h"] = None
            h10_applies = classification.get("h10_global", {}).get("applies", False)

            if h10_applies:
                # 정답이지만 근거 부족 → H10 피드백만 짧게
                feedback = generate_feedback(
                    h_code="H10",
                    observed_errors=grading.get("observed_errors", []),
                    student_context=solution_text if grading_mode == "text" else steps_str,
                )
                print(f"  [분류] 정답 — H10 전역 판정 적용")
            else:
                feedback = {"feedback": "", "detailed_feedback": "", "next_focus": []}
                print(f"  [분류] 정답 — H코드 해당 없음")
        else:
            # LLM 직접 분류 (메인 방식)
            sol_context = solution_text if grading_mode == "text" else steps_str
            classification = classify_error_direct(grading, problem_text, sol_context)

            fb_h_code = classification.get("primary_h")
            if fb_h_code:
                feedback = generate_feedback(
                    h_code=fb_h_code,
                    observed_errors=grading.get("observed_errors", []),
                    student_context=sol_context,
                )
            else:
                feedback = {"feedback": "", "detailed_feedback": "", "next_focus": []}

            print(f"  [분류] 직접={classification.get('primary_h')}, 보조={classification.get('secondary_h')}")

        # 루브릭 점수 추출
        rubric_scores = grading.get("rubric_scores", [])

        # 감점형: 각 기준 score를 검증하고 total 재계산
        # LLM이 rubric_scores 항목을 문자열로 뱉는 경우가 있어 dict만 처리
        rubric_scores = [rs for rs in rubric_scores if isinstance(rs, dict)]
        grading["rubric_scores"] = rubric_scores  # 정제된 리스트로 교체

        for rs in rubric_scores:
            deductions = [d for d in (rs.get("deductions", []) or []) if isinstance(d, dict)]
            rs["deductions"] = deductions  # 정제된 리스트로 교체
            deduction_sum = sum(d.get("points", 0) for d in deductions)  # 감점은 음수
            calculated = max(0, 25 + deduction_sum)  # 25 + (-감점합) = 남은 점수
            # LLM이 보고한 score와 계산값이 다르면 계산값 우선
            if rs.get("score") != calculated:
                print(f"    [보정] {rs.get('criterion')}: LLM={rs.get('score')}, 계산={calculated}")
                rs["score"] = calculated

        total_score = sum(rs.get("score", 0) for rs in rubric_scores)
        # LLM이 보고한 total과 다르면 계산값 우선
        if grading.get("total_score") != total_score:
            print(f"  [보정] total_score: LLM={grading.get('total_score')}, 계산={total_score}")

        print(f"  [루브릭] {len(rubric_scores)}개 기준, 총점={total_score}/100")
        for rs in rubric_scores:
            deds = rs.get("deductions", [])
            print(f"    {rs.get('criterion')}: {rs.get('score')}/25 (감점 {len(deds)}건)")
            for d in deds:
                print(f"      L{d.get('line_ref',0)} {d.get('points')}: {d.get('reason','')[:40]}")

        # answer_id: 이 채점 결과의 고유 식별자 (시트 병합 키)
        answer_id = f"A-{now_iso().replace(':','').replace('-','').replace('.','')[:15]}-{problem_id or 'X'}"

        result = {
            "success": True,
            "answer_id": answer_id,
            "problem_id": problem_id,
            "is_correct": is_correct,
            "grading_mode": grading_mode,
            "student_final_answer": grading.get("student_final_answer", ""),
            "steps": grading.get("steps", []),
            "rubric_scores": rubric_scores,
            "total_score": total_score,
            "max_score": 100,
            "observed_errors": grading.get("observed_errors", []),
            "classification": classification,
            "feedback": feedback,
            "evidence_confidence": evidence_confidence,
            "evidence_failures": evidence_failures,
            "contradiction_flags": contradiction_flags,
            "review_queue": (
                evidence_confidence == "low"
                or bool(contradiction_flags)
                or bool(classification.get("multi_error_flag"))
            ),
            "attempt_count": attempt_count,
            "reveal_answer": (not is_correct) and attempt_count >= MAX_ATTEMPTS_BEFORE_REVEAL,
            "prompt_version": PROMPT_VERSION,
            "model_grading": MODEL_OPUS,
            "model_feedback": MODEL_HAIKU,
            "timestamp": now_iso(),
        }

        print(f"  [채점 완료] correct={result['is_correct']}, "
              f"primary={classification.get('primary_h')}, "
              f"secondary={classification.get('secondary_h')}, total={total_score}")
        return jsonify(result)

    except Exception as e:
        print(f"  [채점 에러] {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


def generate_feedback(h_code: str, observed_errors: list, student_context: str) -> dict:
    """Claude Haiku로 개인화 피드백 생성 (짧은 + 상세 이중 구조)"""
    try:
        template = FEEDBACK_TEMPLATES.get(h_code, "")
        concepts = MISSING_CONCEPTS.get(h_code, [])
        label = ERROR_LABELS.get(h_code, h_code)
        flow = RECOMMENDED_FLOW.get(h_code, "")

        prompt = p_feedback.build(
            h_code=h_code,
            label=label,
            template=template,
            concepts_str=', '.join(concepts),
            flow=flow,
            observed_errors_str=(chr(10).join('- ' + e for e in observed_errors) if observed_errors else "(관찰 없음)"),
            student_context=student_context,
        )

        response = get_claude().messages.create(
            model=MODEL_HAIKU,
            max_tokens=512,
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
    return jsonify({"status": "ok", "time": now_iso(), "version": "v2+rv2",
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


# ─── Supabase 저장 헬퍼 ───
def save_to_supabase(result_data: dict):
    """채점 결과를 Supabase에 저장 (실패해도 서버는 계속 동작)"""
    sb = get_supabase()
    if not sb:
        print("  [Supabase] 클라이언트 없음, 스킵")
        return

    try:
        cls = result_data.get("classification", {}) or {}
        fb = result_data.get("feedback", {}) or {}
        problem_id = result_data.get("problem_id", "")

        # 1. submissions 저장
        submission_id = None
        try:
            sub_row = {
                "problem_id": problem_id,
                "ocr_text": result_data.get("ocr_text", ""),
                "ocr_steps": result_data.get("solution_steps"),
                "attempt_number": result_data.get("attempt_count", 1),
            }
            sub_resp = sb.table("submissions").insert(sub_row).execute()
            if sub_resp.data:
                submission_id = sub_resp.data[0].get("submission_id")
                print(f"  [Supabase] submission 저장 완료 (id={submission_id})")
        except Exception as e:
            print(f"  [Supabase] submission 저장 실패 (무시): {e}")

        # 2. gradings 저장 (v2.2 확장 필드 포함)
        grading_id = None
        grade_row = {
            "problem_id": problem_id,
            "answer_id": result_data.get("answer_id"),
            "is_correct": result_data.get("is_correct", False),
            "student_answer": result_data.get("student_final_answer", ""),
            "graded_steps": result_data.get("steps"),  # status 6종 + depends_on_line_id 포함
            "observed_errors": result_data.get("observed_errors"),
            "primary_h": cls.get("primary_h"),
            "secondary_h": cls.get("secondary_h"),
            "h10_global": cls.get("h10_global"),
            "multi_error_flag": cls.get("multi_error_flag", False),
            "multi_error_lines": cls.get("multi_error_lines"),
            "feedback_text": fb.get("feedback", ""),
            "detailed_feedback": fb.get("detailed_feedback", ""),
            "next_focus": fb.get("next_focus"),
            "total_score": result_data.get("total_score"),
            "max_score": result_data.get("max_score", 100),
            "grading_mode": result_data.get("grading_mode"),
            "prompt_version": result_data.get("prompt_version"),
            "model_grading": result_data.get("model_grading"),
            "model_feedback": result_data.get("model_feedback"),
            "attempt_count": result_data.get("attempt_count", 1),
            "reveal_answer": result_data.get("reveal_answer", False),
            "evidence_confidence": result_data.get("evidence_confidence", "high"),
            "evidence_failures": result_data.get("evidence_failures"),
            "contradiction_flags": result_data.get("contradiction_flags"),
            "review_queue": result_data.get("review_queue", False),
            "model_version": "v2.2",
        }
        if submission_id:
            grade_row["submission_id"] = submission_id

        grade_resp = sb.table("gradings").insert(grade_row).execute()
        if grade_resp.data:
            grading_id = grade_resp.data[0].get("grading_id")
            print(f"  [Supabase] grading 저장 완료 (id={grading_id})")

        # 3. rubric_scores 저장 (감점형: criterion + deductions JSONB)
        if grading_id and problem_id:
            try:
                rub_resp = sb.table("rubrics").select("*").eq("problem_id", problem_id).order("sort_order").execute()
                rubrics = rub_resp.data if rub_resp.data else []

                rubric_scores_data = result_data.get("rubric_scores", [])

                if rubric_scores_data:
                    for i, score_data in enumerate(rubric_scores_data):
                        row = {
                            "grading_id": grading_id,
                            "rubric_id": rubrics[i]["rubric_id"] if i < len(rubrics) else None,
                            "criterion": score_data.get("criterion", ""),
                            "points": score_data.get("score", 0),
                            "max_points": 25,
                            "deductions": score_data.get("deductions", []),  # JSONB 배열 그대로
                        }
                        sb.table("rubric_scores").insert(row).execute()
                    print(f"  [Supabase] rubric_scores {len(rubric_scores_data)}개 저장 완료")
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
                        "observation_points": p.get("observation_points", ""),  # 관찰 지점
                        "deduction_scale": p.get("deduction_scale", {}),        # 감점 척도 (v2.2)
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


# 학생 결과 저장/조회 API
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


@app.route("/results/export", methods=["GET"])
def export_results_csv():
    """채점 결과를 CSV로 내보내기 (답안ID·prompt_version·모델명·시각 포함)"""
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)

    # 헤더
    writer.writerow([
        "result_id", "answer_id", "problem_id", "student_id", "is_correct",
        "student_final_answer", "total_score", "max_score",
        "primary_h", "secondary_h", "h10_global_applies", "h10_global_reason",
        "observed_errors",
        "feedback", "detailed_feedback",
        "prompt_version", "model_grading", "model_feedback",
        "grading_mode", "created_at",
    ])

    for rid, r in _results_db.items():
        cls = r.get("classification", {}) or {}
        fb = r.get("feedback", {})
        h10 = cls.get("h10_global", {}) or {}
        writer.writerow([
            rid,
            r.get("answer_id", ""),
            r.get("problem_id", ""),
            r.get("student_id", ""),
            r.get("is_correct", ""),
            r.get("student_final_answer", ""),
            r.get("total_score", ""),
            r.get("max_score", 100),
            cls.get("primary_h", ""),
            cls.get("secondary_h", ""),
            h10.get("applies", ""),
            h10.get("reason", ""),
            "; ".join(r.get("observed_errors", [])),
            fb.get("feedback", ""),
            fb.get("detailed_feedback", ""),
            r.get("prompt_version", ""),
            r.get("model_grading", ""),
            r.get("model_feedback", ""),
            r.get("grading_mode", ""),
            r.get("created_at", ""),
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
load_problems_from_supabase()  # DB에 있는 문제도 병합


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
