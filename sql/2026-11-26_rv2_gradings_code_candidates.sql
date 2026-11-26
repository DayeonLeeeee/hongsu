-- rv2_gradings: 프롬프트 1.8(2026-11-26, 오세준 교수님 요청)의 code_candidates 저장용 컬럼
-- Railway Postgres(DATABASE_URL)는 서버가 시작할 때 hongsu/db.py가 자동으로 추가한다 (ADD COLUMN IF NOT EXISTS).
-- Supabase에 저장하는 경우에만 SQL Editor에서 1회 실행. 실행 전에도 서버는 기존 컬럼만으로 저장을 계속한다.

ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS code_candidates jsonb;  -- 복수 코드 후보 [{code, reason}]
