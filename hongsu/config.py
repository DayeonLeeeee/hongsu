# coding: utf-8
"""환경변수와 서버 전역 상수."""
import os

# 저장소 루트 (student.html, admin.html, manifest.json 위치)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

MATHPIX_APP_ID  = os.environ.get("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# 진단 모델은 rv2_grader.RV2_MODEL (환경변수 RV2_MODEL). 아래는 OCR·수정용.
MODEL_HAIKU  = "claude-haiku-4-5-20251001"
MODEL_GEMINI = "gemini-2.5-flash"

# 반복 오답 임계값 (미확정, 임의): 이 횟수 넘게 오답이면 정답 공개
MAX_ATTEMPTS_BEFORE_REVEAL = 3

# Claude 호출 안정성: SDK가 429·5xx·529(과부하)·연결 오류를 지수 백오프로 자동 재시도한다.
# 재시도 횟수와 호출당 타임아웃을 명시해 둔다 (SDK 기본값: 재시도 2회, 타임아웃 600초).
# 최악의 경우 한 요청 = 2회 호출(형식 재시도) × 3번 시도 × 120초 = 720초 → gunicorn --timeout(기본 900)보다 짧아야 한다.
CLAUDE_MAX_RETRIES = int(os.environ.get("CLAUDE_MAX_RETRIES", "2"))
CLAUDE_TIMEOUT_S = float(os.environ.get("CLAUDE_TIMEOUT_S", "120"))
