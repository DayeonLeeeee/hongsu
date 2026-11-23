# coding: utf-8
"""학생 풀이 OCR 서비스: Mathpix(수식) + Gemini Flash(평문 단계) 병렬 호출."""
import base64
import contextvars
import json
from typing import Any, Callable, TypeVar
from concurrent.futures import Future, ThreadPoolExecutor

import requests as http_requests

from hongsu import clients
from hongsu.config import MATHPIX_APP_ID, MATHPIX_APP_KEY
from hongsu.logging_setup import log
from prompts import ocr as p_ocr

T = TypeVar("T")

executor = ThreadPoolExecutor(max_workers=4)


def submit_with_context(fn: Callable[..., T], *args: Any, **kwargs: Any) -> "Future[T]":
    """executor.submit + 현재 컨텍스트(request_id) 복사."""
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


# ─── Mathpix OCR ────
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
                log.info(f"[Mathpix] confidence={conf:.2f}, {len(text)}자")
            return text
        else:
            log.warning(f"[Mathpix 실패] HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        log.warning(f"[Mathpix 실패] {e}")
    return ""


# ─── Gemini Flash OCR (평문) ─────
def call_gemini_ocr(image_b64: str, mime_type: str) -> dict:
    """
    Gemini Flash — 학생 풀이 평문 OCR.
    문제는 관리자가 텍스트로 입력하므로 OCR 대상 아님.
    """
    instruction = p_ocr.build_solution_ocr_prompt()
    try:
        model = clients.get_gemini_model()
        response = model.generate_content(
            [instruction, {"mime_type": mime_type, "data": base64.b64decode(image_b64)}],
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        log.warning(f"[Gemini OCR 실패] {e}")
        return {"steps": []}
