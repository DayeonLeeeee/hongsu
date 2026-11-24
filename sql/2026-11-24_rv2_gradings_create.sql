-- rv2_gradings 테이블 새로 만들기 (새 Supabase 프로젝트용, 1.3 컬럼 포함)
-- Railway Postgres(DATABASE_URL)를 쓰면 서버가 이 테이블을 알아서 만든다 (hongsu/db.py). 이 파일은 Supabase용.
-- Supabase SQL Editor에서 1회 실행. 기존 프로젝트에 컬럼만 더할 때는 2026-11-24_rv2_gradings_add_columns.sql
-- 서버(hongsu/storage.py save_rv2_to_supabase)가 넣는 키와 같다.

CREATE TABLE IF NOT EXISTS rv2_gradings (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at          timestamptz NOT NULL DEFAULT now(),

    -- 요청
    answer_id           text,
    problem_id          text,
    student_id          text,
    attempt_count       integer,
    mode                text,          -- with_codebook / without_codebook
    input_status        text,
    answer_text         text,

    -- 채점 결과
    lines               jsonb,
    main_error_code     text,
    justification_level text,
    math_result_correct boolean,
    review_needed       boolean,
    review_reason       text,
    response_complete   boolean,
    missing_criteria    jsonb,
    validation_issues   jsonb,
    error               text,

    -- 재현용 기록
    prompt_version      text,
    codebook_version    text,
    xlsx_sha8           text,
    allowed_codes       jsonb,
    model               text,
    n_calls             integer,
    usage               jsonb,
    stop_reasons        jsonb,
    raw_responses       jsonb,

    -- grader 1.3
    request_id          text,          -- 서버 로그 검색용 요청 ID
    outcome             text,          -- graded / review / incomplete / not_attempted / unreadable / error
    review_sources      jsonb,         -- review로 간 이유: model / validation / max_tokens / error
    grader_version      text,
    tool_choice         text,          -- tool / auto
    response_models     jsonb,         -- API가 실제로 응답한 모델 ID (호출별)
    extraction          jsonb,         -- 호출별 결과 출처: tool / text / none
    first_issues        jsonb          -- 첫 호출의 정제·검증 사유
);

CREATE INDEX IF NOT EXISTS rv2_gradings_problem_idx ON rv2_gradings (problem_id);
CREATE INDEX IF NOT EXISTS rv2_gradings_student_idx ON rv2_gradings (student_id);
CREATE INDEX IF NOT EXISTS rv2_gradings_created_idx ON rv2_gradings (created_at);

-- 학생 답안이 들어가므로 브라우저용(anon) 키로는 읽고 쓰지 못하게 막는다.
-- 서버는 Railway의 SUPABASE_KEY에 secret(service_role) 키를 넣어 이 제한을 우회한다.
ALTER TABLE rv2_gradings ENABLE ROW LEVEL SECURITY;
