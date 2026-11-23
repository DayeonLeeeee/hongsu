# coding: utf-8
"""풀이 OCR(/ocr)과 자연어 수정(/refine). 학생이 OCR 결과를 확인·수정하는 단계."""
import json

from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

from hongsu import clients
from hongsu import ocr as ocr_service
from hongsu.config import MODEL_HAIKU
from hongsu.logging_setup import log
from hongsu.utils import make_image_block, normalize_plain, now_iso, parse_tool_use
from prompts import refine as p_refine

bp = Blueprint("ocr", __name__)

# 자연어 수정 도구 (tool_use 강제)
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


@bp.route("/ocr", methods=["POST"])
def ocr_endpoint() -> ResponseReturnValue:
    """
    학생 풀이 OCR: Mathpix + Gemini Flash 병렬 실행 → 단계별 평문 텍스트 반환.
    문제는 관리자가 텍스트로 등록하므로 OCR 대상 아님.
    """
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")
        mime_type = data.get("mime_type", "image/jpeg")

        log.info(f"[OCR 요청] 이미지={len(image_b64)//1024}KB")

        # 병렬 실행
        f_mathpix = ocr_service.submit_with_context(ocr_service.call_mathpix, image_b64, mime_type)
        f_gemini  = ocr_service.submit_with_context(ocr_service.call_gemini_ocr, image_b64, mime_type)
        mathpix_result = f_mathpix.result()
        gemini_result  = f_gemini.result()

        log.info(f"→ Mathpix: {mathpix_result[:60]}")
        log.info(f"→ Gemini: {json.dumps(gemini_result, ensure_ascii=False)[:100]}")

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
        log.exception("[OCR 에러]")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/refine", methods=["POST"])
def refine_endpoint() -> ResponseReturnValue:
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

        log.info(f"[수정 요청] instruction={user_instruction[:60]}")

        prompt = p_refine.build(current_text=current_text, user_instruction=user_instruction)

        response = clients.get_claude().messages.create(
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

        log.info(f"[수정 완료] changes={result.get('changes')}, agreement={result.get('image_agreement')}")
        return jsonify(result)

    except Exception as e:
        log.exception("[수정 에러]")
        return jsonify({"success": False, "error": str(e)}), 500
