# coding: utf-8
"""공용 도우미: 시각, 이미지 블록, Claude 응답 파싱, 평문 정리."""
import datetime
import re
from typing import Any

import rv2_grader

def now_iso() -> str:
    return datetime.datetime.now().isoformat()


def parse_tool_use(response: Any) -> dict:
    """Claude 응답에서 구조화된 결과(dict)를 꺼낸다. tool_use 블록 우선, 없으면 텍스트의 JSON.
    파싱 규칙은 rv2_grader.extract_result 하나로 통일한다."""
    parsed, _source = rv2_grader.extract_result(getattr(response, "content", []))
    if not isinstance(parsed, dict):
        raise ValueError("Claude 응답에서 구조화된 결과를 찾지 못함")
    return parsed

def make_image_block(image_b64: str, mime_type: str = "image/jpeg") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": image_b64,
        }
    }


def normalize_plain(text: str) -> str:
    """Gemini가 어쩌다 LaTeX 백슬래시를 뱉으면 평문으로 강제 변환"""
    if not text:
        return ""
    s = text
    # LaTeX \frac → (a/b). 분자·분모가 식이면 괄호로 감싼다: \frac{x-2}{3} → ((x-2)/3)
    #   (예전에는 (x-2/3)이 되어 x − 2/3으로 읽혔다)
    def _frac(m: "re.Match") -> str:
        a, b = (t.strip() for t in m.groups())
        wrap = lambda t: t if re.fullmatch(r"[\w.√^]+", t) else f"({t})"  # noqa: E731
        return f"({wrap(a)}/{wrap(b)})"
    s = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', _frac, s)
    # \sqrt{n} → √n, 식이면 √(x+1)  (√x+1로 두면 √x + 1로 읽힌다)
    s = re.sub(r'\\sqrt\s*\{([^{}]+)\}',
               lambda m: "√" + (m.group(1) if re.fullmatch(r"[\w.^]+", m.group(1).strip()) else f"({m.group(1).strip()})"), s)
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
