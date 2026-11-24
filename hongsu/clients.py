# coding: utf-8
"""외부 API 클라이언트 (Claude, Gemini, Supabase). 처음 쓸 때 한 번 만든다.

라우트·서비스는 `clients.get_claude()`처럼 모듈을 거쳐 호출한다 (테스트에서 가짜로 바꿀 수 있게).
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # 타입 표기용 (실행 시에는 처음 쓸 때 import)
    import anthropic
    import google.generativeai as genai
    from supabase import Client

from hongsu.config import (ANTHROPIC_API_KEY, CLAUDE_MAX_RETRIES, CLAUDE_TIMEOUT_S, GEMINI_API_KEY,
                           MODEL_GEMINI, SUPABASE_KEY, SUPABASE_URL)
from hongsu.logging_setup import log

claude_client = None
_gemini_configured = False


def get_claude() -> "anthropic.Anthropic":
    global claude_client
    if claude_client is None:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY,
                                            max_retries=CLAUDE_MAX_RETRIES,
                                            timeout=CLAUDE_TIMEOUT_S)
    return claude_client


def get_gemini_model() -> "genai.GenerativeModel":
    global _gemini_configured
    import google.generativeai as genai
    if not _gemini_configured:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_configured = True
    return genai.GenerativeModel(MODEL_GEMINI)


_supabase = None
def get_supabase() -> Optional["Client"]:
    global _supabase
    if _supabase is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            log.info("[Supabase] 연결 성공")
        except Exception as e:
            log.warning(f"[Supabase] 연결 실패: {e}")
    return _supabase
