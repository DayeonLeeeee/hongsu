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

# 채점 결과 저장소: DATABASE_URL(PostgreSQL, 예: Railway Postgres)이 있으면 그쪽, 없으면 Supabase
DATABASE_URL = os.environ.get("DATABASE_URL", "")

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

# 입력 상한 (hongsu/guard.py). 합성 답안은 500자 안팎, 학생 풀이도 수십 줄을 넘기 어렵다.
MAX_ANSWER_CHARS = int(os.environ.get("MAX_ANSWER_CHARS", "5000"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60"))
MAX_REQUEST_MB = float(os.environ.get("MAX_REQUEST_MB", "20"))      # 이미지(base64) 포함 요청 본문
# 요청 횟수 제한 (1분당, 엔드포인트별). 0이면 끈다.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))        # 학생(student_id)별
# 관리 기능(문항 추가·수정·삭제, 엑셀 다시 읽기, 사용량)을 막는 토큰. 설정하면 요청 헤더 X-Admin-Token이 같아야 한다.
# 비워 두면 예전처럼 열려 있다 (지금 admin.html은 토큰을 보내지 않는다).
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
RATE_LIMIT_IP_PER_MIN = int(os.environ.get("RATE_LIMIT_IP_PER_MIN", "600"))  # IP별 (교실 = 공유기 IP 하나)
