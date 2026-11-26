# 변경 이력

채점 결과에 영향을 주는 파일별로 정리한다. 결과 파일·DB의 `prompt_version`, `grader_version`, `git_commit`으로 어느 버전으로 채점했는지 찾는다.

- 프롬프트를 바꾸면 `prompts/rv2.py`의 `PORT_VERSION`을 올린다. 안 올리면 `tests/test_units.py`의 프롬프트 지문 테스트가 실패한다.
- 정제·검증·재시도 로직을 바꾸면 `rv2_grader.py`의 `GRADER_VERSION`을 올린다.
- 둘 중 하나라도 바꾸면 `experiments/regression.py`(실험 3-3)로 고정 답안 20건을 다시 돌려 기준과 비교한다.

## prompts/rv2.py — 채점 프롬프트 (`PORT_VERSION`)

| 버전 | 날짜 | 변경 | 판정 기준 변경 |
| --- | --- | --- | --- |
| hongsu-rv2-1.8 | 2026-11-26 | 오세준 교수님 11/26 답변(미완 기준) 확인 1)·2): 판정 절차 4번에서 "그 누락이 기준 판정표의 'wrong일 때'에 해당하는 오류이면 그 코드를 main_error_code에 적고" 삭제(누락·중단은 missing_criteria와 response_complete=false로만), `main_error_code` 설명에서 "(wrong 줄이 없으면 오류인 누락 기준)" 삭제, 출력에 `code_candidates`([{code, reason}], main_error_code 줄에 함께 성립한 다른 후보, 없으면 []) 추가. C-1은 그대로 | 있음 (누락 기준에 코드 안 붙임) |
| hongsu-rv2-1.7 | 2026-11-26 | 코드북 R v3.1 자료(`materials/dataset_rv31.xlsx`)를 읽을 수 있게 함: 코드 정의에 '주의' 줄(`caution` 열이 있을 때만), 판정 절차 4의 미완 처리 규칙 참조를 자료의 규칙 번호에 맞춤(v3.1이면 '판정 규칙 답안 상태', 아니면 '판정 규칙 12'). R-v2/R-v3 자료(`dataset_rv2.xlsx`)로 만들면 1.6과 글자 하나 다르지 않음. 판정 절차 C-1과 4번 '누락이 오류이면…' 문장은 교수님 답 기다리며 그대로(v3.1 규칙 12와 충돌할 수 있음) | 자료를 v3.1로 바꾸면 있음 |
| hongsu-rv2-1.6 | 2026-11-26 | 1.3 + 줄마다 `violated_criterion`(wrong이면 그 줄의 오류가 어긴 기준 번호, 아니면 null). 1.4·1.5의 `error_criterion`과 같은 칸인데, 이름을 코드북 R v3.1 기록 표(`line_results[].violated_criterion`)에 맞춤. 코드북의 `error_criterion`(이 줄이 어느 기준에 해당하는가)은 여기의 `matched_criterion`. `stopped_midway` 칸은 넣지 않음(1.5에만 있음) | 없음 (출력 칸 추가) |
| hongsu-rv2-1.5 | 2026-11-26 | 답안 전체에 `stopped_midway`(학생이 끝까지 쓰지 않고 중간에 멈춰 최종 답을 내지 않았으면 true, 끝까지 쓰고 답을 냈으면 요구 항목이 빠졌거나 틀렸어도 false) 칸 추가. 11/11 회의 합의 3(정오답은 끝까지 썼는지로 나누고 중간에 멈춘 답안은 미완). `response_complete`(요구 항목 누락)는 그대로 | 없음 (출력 칸 추가) |
| hongsu-rv2-1.4 | 2026-11-26 | 줄마다 `error_criterion`(wrong이면 그 줄의 오류가 어긴 기준 번호, 아니면 null) 칸 추가. `matched_criterion`(그 줄이 해당하는 기준)은 그대로. 11/11 회의 합의 2(줄을 기준에 맞추는 일과 틀린 곳 진단을 한 칸에 섞지 않음). 출력 형식 규칙의 정수 필드 목록에도 추가. 판정 절차·규칙 문구는 그대로 | 없음 (출력 칸 추가) |
| hongsu-rv2-1.3 | 2026-11-24 | 스키마 뒤에 출력 형식 규칙(`FORMAT_RULES`) 추가: 줄은 객체로, 불리언·null은 문자열 금지, judgment 소문자, 번호 필드는 정수, 도구를 못 쓰면 ```json 블록 하나 | 없음 |
| hongsu-rv2-1.2 | 2026-10~11 | 코드북 R-v3(2026-10-30 오세준 교수님 승인본) 대응: `response_complete`·`missing_criteria` 출력 추가(규칙 12), '안 쓴 기준' 처리 단계 추가, C-3 적용 범위를 결론이 옳은 답안으로 한정(규칙 11), 코드북 버전을 엑셀에서 읽음 | 있음 (R-v3) |
| (이식 초판) | 2026-10 | 키트 `prompt_builder.py`(kit-1.2-20261022) 이식. 자료 출처만 엑셀 직접 로드로, 출력은 tool_use로 | 키트와 동일 |

**1.3에서 일부러 뺀 규칙 (주석으로 보존).** 아래 두 문장은 줄 나누는 방식, 곧 줄별 판정 내용을 바꿀 수 있어 넣지 않았다. `prompts/rv2.py`의 `FORMAT_RULES` 위 주석에 그대로 남아 있고, 되돌리려면 `"- lines의 각 원소는 …"` 다음 줄에 넣고 `PORT_VERSION`을 올린다.

- `학생 답안의 줄 하나 = lines 원소 하나.`
- `여러 줄을 한 원소에 합치지 않는다.` (원래 "lines의 각 원소는 …" 규칙 끝에 붙어 있던 문장)

## materials/ — 채점 자료 (`codebook_version`)

| 자료 | 날짜 | 변경 |
| --- | --- | --- |
| dataset_rv31.xlsx (R-v3.1) | 2026-11-26 | `experiments/build_dataset_v31.py`가 키트 `dataset_rv2.xlsx`(R-v3, 원본은 그대로)를 복사해 오세준 교수님 「오류 코드북 R v3.1」(2026-11-26)을 반영. 바꾼 칸은 전부 시트 `CHANGES_v3.1`에 기록. 코드 표(R2·R5·R6·R9·R10·R11 정의·포함·제외·주의, R4는 1차 보류 표시), 판정 규칙 11 단서·규칙 12 신설(옛 12는 '답안 상태'로 개명), 문항 2 ④·문항 7 ③④ 기준표, 문항 7 주 코드 R2. 문항 1 수치는 반영 안 함(2차 적용부터). 같은 날 교수님 답변(미완 기준)으로 규칙 번호 수정: 12 = Rv3 미완 처리 그대로, 주 코드 선정 = 13(교수님 v3.2 예정), 규칙 11 단서와 R5 제외 문구의 참조도 13으로 |

## rv2_grader.py — 응답 정제·검증·재시도 (`GRADER_VERSION`)

| 버전 | 날짜 | 변경 |
| --- | --- | --- |
| rv2-grader-1.8 | 2026-11-26 | `code_candidates` 형식 정리(문자열·JSON 문자열도 [{code}]로, 코드는 대문자). 검증 규칙은 그대로 |
| rv2-grader-1.7 | 2026-11-26 | `violated_criterion`을 정수로 정리(1.4·1.5 결과의 `error_criterion`도 계속 정리). 검증 규칙은 그대로 |
| rv2-grader-1.6 | 2026-11-26 | `stopped_midway`를 불리언으로 정리(빠지면 null, 추정하지 않음). 검증 규칙은 그대로(미완 R3 재시도 규칙은 계속 `response_complete` 기준). 키트 방식 결과 분류(`outcome`의 incomplete, 실험의 미완 회차)도 그대로 `response_complete` 기준 |
| rv2-grader-1.5 | 2026-11-26 | `error_criterion`을 `matched_criterion`과 같은 방식으로 정수 정리(①, "3" → 정수). 검증 규칙은 추가하지 않음 |
| rv2-grader-1.4 | 2026-11-25 | 검증 규칙 1개 추가: 미완 답안(response_complete=false)에 주 코드 R3가 붙으면 사유를 붙여 재시도하고, 그래도 남으면 검수로 보낸다(계획 문서 라벨·데이터 작업 2번). 1.3 결과 269회 중 해당 사례 0건이라 지금까지 결과에는 영향이 없다. `/grade` 응답에 `completion_status` 추가, 미완이면 `is_correct`는 null |
| rv2-grader-1.3 | 2026-11-24 | 응답 정제를 `sanitize()` 하나로 통합(고칠 수 있는 형식은 고치고, 못 고치는 것은 사유를 남겨 1회 재시도·검수). tool_use 블록이 없으면 텍스트의 JSON을 읽음(`extract_result`). 결과 블록이 없어도 1회 재시도. 모델별 호출 기본값(Opus 5.5: 도구 auto, temperature 없음). 기본 모델 Opus 5.5. `outcome`(graded/review/incomplete/not_attempted/unreadable/error)과 `review_sources` 분류. 기록 추가: `extraction`, `first_issues`, `response_models`, `tool_choice`, `git_commit` |
| (1.2 이전) | 2026-10 | 키트 `run_eval.validate()` 이식: 인용문 원문 대조, 허용 밖 코드 거부, 실패 사유를 붙여 1회 재시도, `missing_criteria` 번호 검사 |

1.3 정제 규칙 중 판정에 닿을 수 있는 것은 모두 보수적으로 정했다: 인용(quote)·근거(reason)의 "없음", "-"는 학생이 실제로 쓴 글자일 수 있어 null로 바꾸지 않는다. judgment의 별칭(정답/오답 같은 말)은 해석하지 않고 재시도로 보낸다. `depends_on` 목록은 자르지 않는다. 소수 줄 번호는 버리지 않고 해석 불가로 둔다.

## 서버 (hongsu/) — 채점 결과에는 영향 없음

| 날짜 | 변경 |
| --- | --- |
| 2026-11-26 | `/grade`의 정답 여부·미완 표시를 11/11 합의 3에 맞춤: `is_correct`는 `stopped_midway`가 true일 때만 비우고(중간에 멈춘 답안), 끝까지 쓴 오답은 오답으로 둠. `completion_status`도 `stopped_midway` 기준. 11/25에 `response_complete`(요구 항목 누락)로 비우던 것은 끝까지 쓴 오답(A-05-B 등)까지 비워서 바꿈 |
| 2026-11-25 | 3회 점검 수정: `answer_id`에 무작위 6자를 붙여 같은 초에 같은 문항을 낸 학생끼리 겹치지 않게 함. 요청 횟수 제한의 IP를 Railway가 덮어쓰는 `X-Real-IP` 기준으로(클라이언트가 꾸밀 수 있는 `X-Forwarded-For` 대신). DB 저장 실패 시 기록을 버리지 않고 보류했다가 다시 넣음(`/health?deep=1`에 `pending`). `ADMIN_TOKEN`을 설정하면 관리 기능(문항 추가·수정·삭제, 엑셀 다시 읽기, 사용량)에 `X-Admin-Token` 필요 |
| 2026-11-25 | API 사용량·비용: 채점마다 `cost_usd`(호출별 실제 응답 모델 가격, `usage_cost.py`) 기록, 로그에 토큰·비용, `/refine`(Haiku)·`/ocr`(Gemini 토큰 + Mathpix 요청당) 비용, `GET /admin/usage`(서버 누적 + DB 날짜·모델별 합계), `experiments/usage_report.py` |
| 2026-11-25 | 입력 검증(잘못된 요청은 400과 이유, 답안·단계 수·본문 크기 상한), 요청 횟수 제한(학생별·IP별 1분당, 비용 드는 엔드포인트만), 결과에 `git_commit` 기록, requirements 버전 고정 |
| 2026-11-25 | OCR 평문 정리 버그 수정: `\frac{x-2}{3}`이 `(x-2/3)`, `\sqrt{x+1}`이 `√x+1`로 바뀌던 것을 `((x-2)/3)`, `√(x+1)`로 (`/ocr`, `/refine` 텍스트만) |
| 2026-11-25 | 타입 힌트, 문항·기준표 조회를 `prompts/rv2.py` 하나로 통일 |
| 2026-11-24 | 채점 결과 저장을 Railway PostgreSQL로(`DATABASE_URL`, 없으면 Supabase). 학생이 본 것(`reveal_answer`, `feedback`) 저장 |
| 2026-11-24 | `hongsu_server.py` 단일 파일을 `hongsu/` 패키지로 분리(동작 동일). 로깅과 요청별 `request_id`. `/health?deep=1`. `/measure_consistency` 중첩 요청 오류 수정 |
