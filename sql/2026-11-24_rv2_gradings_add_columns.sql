-- rv2_gradings: 안정화 작업(grader 1.3)에서 추가된 결과 필드 저장용 컬럼
-- Supabase SQL Editor에서 1회 실행. 이미 있으면 건너뛴다 (IF NOT EXISTS).
-- 실행 전에 배포해도 서버는 기존 컬럼만으로 저장을 계속한다 (save_rv2_to_supabase 대비책).

ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS request_id      text;   -- 서버 로그 검색용 요청 ID
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS outcome         text;   -- graded / review / incomplete / not_attempted / unreadable / error
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS review_sources  jsonb;  -- review로 간 이유: model / validation / max_tokens / error
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS grader_version  text;   -- 채점 로직(정제·검증·재시도) 버전
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS tool_choice     text;   -- tool / auto
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS response_models jsonb;  -- API가 실제로 응답한 모델 ID (호출별)
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS extraction      jsonb;  -- 호출별 결과 출처: tool / text / none
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS first_issues    jsonb;  -- 첫 호출의 정제·검증 사유
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS reveal_answer   boolean; -- 학생에게 정답을 공개했는지
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS feedback        jsonb;  -- 학생에게 보여 준 피드백
ALTER TABLE rv2_gradings ADD COLUMN IF NOT EXISTS git_commit      text;   -- 채점에 쓴 코드의 git 커밋
