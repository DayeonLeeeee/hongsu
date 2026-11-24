# 실험 실행 안내 — 교수님 점검 전에 먼저 하는 실험

교수님 점검(다음 미팅) 없이 우리끼리 먼저 하기로 한 실험이다.

- **먼저 바꾼 것 검증**: 교수님 확인 없이 먼저 적용한 코드 변경(hongsu 1.3 안정화, 채점 모델 Opus 5.5)이 결과를 바꾸지 않았는지 → 1-1
- **코드 개선 후 재확인**: 교수님께 바로 여쭙지 않고, 코드를 고친 뒤에도 같은 현상이면 그때 건의할 답안 → 1-9
- **미팅에 가져갈 숫자**: 교수님께 여쭐 질문의 근거 → 1-2 ~ 1-8

미팅에서 결정을 받은 뒤에 하는 실험(확신도 기준선, 기권 차단 프롬프트, 캐스케이드, 주 코드 선정 규칙)은 여기 없다.

**모든 실험은 `experiments/run.py` 하나로 돌린다. Windows와 macOS에서 명령이 같다** (macOS는 `python` 대신 `python3`).

---

## 0. 준비 (처음 한 번)

### Windows (PowerShell)
```powershell
cd <hongsu 폴더>
git checkout stabilize
git pull
python -m pip install -r requirements.txt pytest
python -m pytest tests -q

$env:PYTHONUTF8 = "1"                          # 창을 새로 열 때마다 (한글 cp949 오류 방지)
$env:ANTHROPIC_API_KEY = "sk-ant-..."         # 새 채점(1-1b, 1-4)할 때만

python experiments/run.py setup --kit "C:\Users\USER\Desktop\연구\rv2_kit"
python experiments/run.py status
```

### macOS (터미널, zsh)
```bash
cd <hongsu 폴더>
git checkout stabilize
git pull
python3 -m venv .venv && source .venv/bin/activate   # 처음 한 번. 이후 터미널을 열 때마다 source만
python3 -m pip install -r requirements.txt pytest
python3 -m pytest tests -q

export ANTHROPIC_API_KEY="sk-ant-..."                # 새 채점(1-1b, 1-4)할 때만

python3 experiments/run.py setup --kit ~/Desktop/연구/rv2_kit   # 키트를 둔 실제 위치로
python3 experiments/run.py status
```

- `setup`은 키트 위치를 `answers/.kit_path`에 기억한다. 이후 명령에는 `--kit`을 안 줘도 된다 (각자 PC에만 있고 저장소에는 안 올라간다).
- `setup`이 만드는 답안 파일 (이미 있으면 덮어쓰지 않는다. 새로 만들려면 `--force`, 고친 라벨이 지워지니 주의):

| 파일 | 내용 | 저장소 |
| --- | --- | --- |
| `answers/expert_33.csv` | 키트 복사본 + `전문가_첫오류_통일` 열 (1-2용, 처음엔 원래 라벨 복사) | 올려도 됨 |
| `answers/holdout45.csv` | 키트 복사본 | 안 올라감 (holdout 봉인) |
| `answers/converted_5.csv` | 1-4용 틀 (`원래_답안`에 원문, `학생_답안` 빈칸) | 올려도 됨 |

- 키트 원본 파일은 건드리지 않는다. 라벨 수정은 `answers/` 사본에서만 한다.
- 키트는 macOS 사용자도 같은 폴더 구조(`rv2_kit/answers`, `rv2_kit/results`)로 받아 두면 된다. 11/11 결과 파일 두 개(`stage3_expert_33_opus55_20261111_1422.jsonl`, `stage3_holdout45_opus55_20261111_1531.jsonl`)가 `results/`에 있어야 한다.

`status`는 지금 돌릴 수 있는 실험과 사람이 할 일(라벨 몇 건 고쳤는지, 변환 답안 몇 건 썼는지, 손글씨 몇 장인지)을 보여 준다.

---

## 실험 한눈에

| 명령 | 실험 | 비용 | 사람이 먼저 할 일 | 상태 (11/24) |
| --- | --- | --- | --- | --- |
| `run.py baseline` | 1-1a·1-3·1-5·1-6·1-7·1-9 (11/11 결과) | 0 | 없음 | **완료** |
| `run.py 1-2` | 라벨 통일 후 첫 오류 줄 | 0 | 라벨 4건 수정 | 라벨 수정 대기 |
| `run.py 1-1b` | 1.3 재채점 + 1.2와 비교 + 1-9 재확인 | 약 219회 호출 | 없음 | 실행 대기 |
| `run.py 1-4` | 설명문형 5건 변환 | 약 15회 호출 | 5건 다시 쓰기 | 작성 대기 |
| `run.py 1-8` | 손글씨 OCR 비교 | OCR API (+`--grade`면 채점) | 손글씨 30~50장 + 정답 타이핑 | 수집 대기 |

보고서는 `experiments/out/<명령>_<날짜_시각>.md`로 쌓인다. 새 채점 결과는 `results/`에 쌓인다.
비용이 드는 명령은 호출 수를 보여 주고 `y`를 입력해야 진행한다 (`--yes`로 건너뜀).
새로 채점하지 않고 `results/`의 최신 결과로 분석만 다시 하려면 `--reuse`.

### 권장 순서
1. 1-8 손글씨 수집 시작 (가장 오래 걸림)
2. `run.py 1-2` — 라벨 4건 수정 후
3. `run.py 1-1b` — 재채점, 1.2와 비교, 1-9 확인
4. `run.py 1-4` — 변환 5건 작성 후
5. `run.py 1-8`

---

## 기준 결과 파일 — 11/10·11/11 최신본만 쓴다

| 파일 (키트 `results/`) | 모델 | 답안 | 반복 | 쓰는 곳 |
| --- | --- | --- | --- | --- |
| `stage3_expert_33_opus55_20261111_1422.jsonl` | Opus 5.5 | expert 33 | 3 | baseline, 1-2, 1-1b 비교, 1-4 변환 전 |
| `stage3_holdout45_opus55_20261111_1531.jsonl` | Opus 5.5 | holdout 40 | 3 | baseline, 1-1b 비교 |
| `stage3_expert_33_server_server_20261111_0859_haiku45.jsonl` 외 | Haiku 4.5·5.5, Opus 4.6 | expert / holdout | 3 | 모델 비교용 (선택) |

쓰지 않는 파일: 10/27 결과, 11/10 00:46~01:00 세 파일(키트 단독 실행, 33건 × 1회, 코드북 버전 기록 없음), `.jsonl.jsonl`로 끝나는 중복본.
11/11 기준 파일은 프롬프트 hongsu-rv2-1.2, 코드북 R-v3이다. 1.3 결과와 같은 표에 섞지 않는다 (비교는 1-1b 비교 보고서로).

---

## 실험별 내용

### baseline — 1-1a · 1-3 · 1-5 · 1-6 · 1-7 · 1-9 (완료, 비용 0)
```
python experiments/run.py baseline
```
11/11 결과로 expert·holdout 보고서를 만든다. 11/24 결과:

- **1-1a 스키마 안정성 (1.2)**: Opus 5.5(도구 강제 끔) 219번 호출 모두 도구 응답, 재시도·에러·남은 검증 문제 0건
- **1-3 A-05-B·A-07-B**: 코드는 3회 모두 맞음(R6, R2). 첫 오류 줄만 라벨과 다름(1 vs 3, 3 vs 4) → 라벨 번호 문제로 보임
- **1-5 기권 (holdout 40건, 답안 단위 다수결)**

  | 기준 | 일치 |
  | --- | --- |
  | 팀 채점기 기준 (기권·코드누락 = 불일치) | 31/40 |
  | '코드 없음' 라벨에 대한 기권을 일치로 인정 | 35/40 |
  | 기권 + 코드누락까지 인정 | 36/40 (11/11 자료의 36/40) |

  진짜 손실(라벨에 코드가 있는데 기권)은 **H-15 1건**. 보고서의 기권 목록이 이용오 교수님 재문의에 넣을 답안 ID다.
- **1-6 미완**: 라벨이 미완인 답안(H-37·H-43, A-08-A·A-08-D)은 AI도 3회 모두 미완. 미완인데 R3가 붙은 경우 0건. 다만 holdout에서 코드가 붙은 답안 5건(H-12, H-18, H-25, H-30, H-45)도 AI가 한 번 이상 `response_complete=false`로 표시했다. 코드북의 `response_complete`는 '요구 기준을 다 다뤘나'이고 담론 2번의 미완은 '중간에 멈췄나'라서, 미완 처리 규칙을 정하기 전에 정리가 필요하다.
- **1-7 R6**: R6 라벨 3건 중 H-15만 기권. 모델 사유: "R6은 y에 대해 정리하는 과정의 이항·부호 오류로 정의돼 있는데, 이 학생은 정리는 옳게 하고 상수항 부호를 잘못 읽었다" → R6 정의 범위 질문의 근거
- **1-9 (11/11 기준)**: 보고서의 '지정 답안 확인' 섹션. 아래 1-1b 참고

### 1-2 라벨 통일 후 첫 오류 줄 (비용 0)
1. `answers/expert_33.csv`의 `전문가_첫오류_통일` 열에서 아래를 **기준표 번호**(엑셀 `06_LINE_JUDGMENT`의 `line_no`)로 고친다. `①`이든 `1`이든 된다. 원래 `전문가_첫오류` 열은 그대로 둔다.
   - 고칠 답안: **A-01-A, A-01-B, A-04-A, A-04-B** (11/11 자료 C-1)
   - 확인할 답안: **A-05-B, A-07-B**
2. 실행:
```
python experiments/run.py 1-2
```
11/11 결과로 통일 라벨 기준 일치율을 낸다 (현재 라벨 그대로면 9/17). 1-1b를 돌렸으면 1.3 결과로도 같이 낸다.
라벨은 기준표를 보고 사람이 정한다. 보고서의 'AI 첫 줄'을 보고 라벨을 맞추면 평가 데이터가 오염된다.

### 1-1b 1.3 재채점 검증 + 1-9 재확인 (약 219회 호출)
교수님 확인 없이 먼저 적용한 1.3 변경(출력 형식 규칙 추가, 재시도 조건 확대)이 판정을 바꾸지 않았는지 확인한다.
```
python experiments/run.py 1-1b
```
하는 일: 현재 코드(Opus 5.5, grader 1.3)로 expert 33건 × 3회, holdout 40건 × 3회 채점 → 보고서 4개

| 보고서 | 볼 것 |
| --- | --- |
| `1-1b_expert33_1.3_*`, `1-1b_holdout45_1.3_*` | 1-1 섹션: 1.3의 형식 문제·재시도 사유 / 지정 답안 확인 섹션 = **1-9** |
| `1-1b_expert33_compare_*`, `1-1b_holdout45_compare_*` | 맨 아래 "다수결 판정이 달라진 답안" 수. 0~1건이면 1.3 변경이 판정에 영향 없다고 말할 수 있다 |

**1-9 코드 개선 후 재확인** — 1.3 보고서의 '지정 답안 확인'을 baseline 보고서의 같은 섹션과 나란히 본다. 다수결 코드와 오류 줄이 그대로면 '같은 현상'.

| 답안 | 현상 (11/11) | 1.3에서도 같으면 |
| --- | --- | --- |
| H-03, H-33 | 오류가 두 개인 답안에서 모델은 앞 줄, 라벨은 뒤 줄 코드를 고름 | 주 코드 선정 기준을 교수님께 여쭘 (오세준 교수님 견해 R1) |
| H-10, A-07-B | 모델이 기준표 문구를 따라 라벨과 다른 유형을 고름 | 기준표 점검 요청 |
| H-15 | 부호 오류를 R6로 보지 않고 기권 | R6 정의 범위 수정 건의 |

비용 감: 11/11 자료 기준 Opus 5.5 holdout 3회 약 $6.4, expert 3회도 비슷한 규모.
채점이 중간에 끊기면 다시 `run.py 1-1b`를 돌린다 (새 결과 파일로 처음부터). 분석만 다시 하려면 `run.py 1-1b --reuse`.

### 1-4 설명문형 5건 변환 (약 15회 호출)
1. `answers/converted_5.csv`의 `학생_답안` 5칸을 풀이 형태(`① … ② …`, 11/11 자료의 A-06-C 예시처럼)로 채운다. 대상: A-05-D, A-06-A, A-06-C, A-08-A, A-09-B. 빈칸은 건너뛴다.
2. 실행:
```
python experiments/run.py 1-4
```
변환 전(11/11 결과)과 변환 후(새 채점)를 답안별로 비교한다. 변환 전은 1.2, 변환 후는 1.3 코드다. 1-1b에서 판정 차이가 있었으면 `run.py 1-4 --reuse --before-13`으로 1.3 결과끼리 다시 비교한다.

### 1-8 손글씨 OCR 비교 (OCR API 비용)
1. 손글씨 **30~50장** (고1 수준. 지인 학생, 섭외 학교 한 반, 안 되면 팀원이 일부러 대충 쓴 것)
   - `experiments/templates/ocr_samples.csv`를 `answers/ocr/ocr_samples.csv`로 복사해서 채운다. 사진은 `answers/ocr/images/`에. `answers/ocr/`는 저장소에 안 올라간다
   - `정답_텍스트`: 학생이 쓴 그대로, **틀린 값도 그대로**, 줄은 줄바꿈으로
   - 일부는 일부러 틀린 값을 쓰게 하고 그 식을 `교정검사` 열에 (예: `f(-2)=5`)
   - 한글 서술 줄 / 수식만 / 섞인 줄이 골고루
2. 엔진 키 설정 (없는 엔진은 건너뛰고 `키 없음`으로 표시):

   | 엔진 | Windows | macOS |
   | --- | --- | --- |
   | Gemini | `$env:GEMINI_API_KEY = "..."` | `export GEMINI_API_KEY="..."` |
   | Mathpix | `$env:MATHPIX_APP_ID = "..."` / `$env:MATHPIX_APP_KEY = "..."` | `export MATHPIX_APP_ID="..."` / `export MATHPIX_APP_KEY="..."` |
   | Azure | `$env:AZURE_VISION_ENDPOINT = "https://<이름>.cognitiveservices.azure.com"` / `$env:AZURE_VISION_KEY = "..."` | `export AZURE_VISION_ENDPOINT="..."` / `export AZURE_VISION_KEY="..."` |
   | Claude | `ANTHROPIC_API_KEY` (위와 같음), 모델은 `OCR_CLAUDE_MODEL`로 변경 가능 (기본 Opus 5.5) | 같음 |
3. 실행:
```
python experiments/run.py 1-8                         # 인식률·교정 위험
python experiments/run.py 1-8 --grade                 # + 채점 영향 (채점 비용 추가)
python experiments/run.py 1-8 --engines gemini,claude # 일부 엔진만
```

---

## 보고서 읽는 법 / 정의

| 섹션 | 볼 것 |
| --- | --- |
| 1-1 | 도구/텍스트/없음 호출 수, 재시도·에러 비율. 1.3 결과면 첫 호출 사유(형식·검증)도 |
| 1-2 | 첫 오류 줄 일치 (1회차 기준) + 회차별 + 답안별 표 |
| 지정 답안 확인 (1-3, 1-9) | 지정 답안의 회차별 코드·다수결·코드 일치, 1회차 오류 줄(학생 줄→기준 번호:코드)과 모델 사유 |
| 1-5 | 세 기준의 일치율, 기권·코드누락 답안 목록 (모델 사유 포함) |
| 1-6 | 라벨 미완 답안의 AI 판단, AI가 미완으로 본 답안, 미완인데 R3가 붙은 회차 |
| 1-7 | 지정 코드 라벨 답안의 AI 판정과 사유, 라벨×AI 교차표 |
| compare | 답안별 전/후 판정, 다수결이 달라진 답안 수 (양쪽 결과가 다 있는 답안만) |

일치율 정의는 팀 채점기 `experiments/score_rv2.py`(키트 `rv2_kit/score_rv2.py` 복사본)를 그대로 쓴다.
- 답안별 주 코드 = 반복 다수결. 기권(`review_needed`)·코드누락(wrong 줄은 있는데 코드 없음)은 표에서 빼고, 남는 표가 없으면 그대로 기권/코드누락
- 라벨이 '코드 없음'이면 AI가 오류없음·미완일 때만 일치
- 첫 오류 줄 = 첫 wrong 줄의 `matched_criterion` (기준표 번호), 1회차 결과
- holdout: 상태 '유효' 40건, H-09·H-13 라벨은 R-v3에 따라 R2로 집계

---

## 개별 스크립트 (직접 옵션을 줄 때)

`run.py`가 아래를 대신 부른다. 다른 결과 파일(예: 다른 모델의 11/11 결과)을 분석하려면 직접 쓴다.

| 스크립트 | 하는 일 | 예 |
| --- | --- | --- |
| `score_experiments.py` | 결과 1개 분석 (1-1, 1-2, 지정 답안, 1-5, 1-6, 1-7) | `--results <jsonl> --answers answers/holdout45.csv --only 1-5,focus` |
| `compare_runs.py` | 결과 2개를 답안별로 비교 | `--before <jsonl> --after <jsonl> --before-answers ... --after-answers ... --ids all` |
| `ocr_compare.py` | OCR 엔진 비교 | `--samples answers/ocr/ocr_samples.csv --engines gemini,claude` |
| `prepare_answers.py` | 답안 파일 준비 | `--kit <rv2_kit>` |
| `../run_batch.py` | 채점 (API 호출) | `--answers answers/expert_33.csv --repeats 3` |

`score_experiments.py`의 주요 옵션: `--first-col`(첫 오류 라벨 열 바꾸기), `--focus-ids`(지정 답안, 빈 문자열이면 생략), `--focus-code`(1-7 코드, 기본 R6), `--latest`(여러 파일·`*` 패턴 중 최신 하나만), `--no-v3-relabel`, `--only`, `--out`.
`*` 패턴은 따옴표로 감싼다 (macOS zsh는 따옴표 없이 쓰면 패턴이 안 맞을 때 에러가 난다).

## 알아둘 점

- `experiments/score_rv2.py`는 키트 원본 복사본이다. 정의를 바꾸려면 키트 원본과 같이 바꾼다. 교수님께 드린 자료는 채점 결과·이유를 정리한 엑셀이고 이 스크립트 출력과는 별개다.
- 보고서 첫 표에 모델·프롬프트·grader·코드북 버전이 나온다. **버전이 섞인 결과끼리는 같은 표에서 비교하지 않는다.**
- 1.2 결과에는 `first_issues` 기록이 없어, 1-1a의 결과 출처는 응답 원문 블록 종류로 셌다.
- Azure 연동은 실제 키로 아직 확인하지 않았다. 첫 실행 때 `experiments/out/1-8_ocr_*_raw.jsonl`의 azure 행을 본다.
- 1-8은 엔진 단독 비교다. 여러 엔진을 합치는 대조 단계는 아직 없다. MyScript는 펜 획 좌표가 필요해서 사진으로는 비교할 수 없다.
- 테스트: `python -m pytest tests -q` (macOS: `python3 -m pytest tests -q`)
