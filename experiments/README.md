# 미팅 전 필수 실험 실행 안내

미팅 전에 숫자를 들고 가야 하는 실험 1-1 ~ 1-8을 돌리는 방법이다.
채점은 기존 `run_batch.py`로 하고, 분석은 이 폴더의 스크립트로 한다. 분석 스크립트는 LLM을 부르지 않는다.

| 실험 | 무엇을 보나 | 근거로 쓰는 질문 | 필요한 것 |
| --- | --- | --- | --- |
| 1-1 스키마 안정성 | 응답 형식이 깨지는 비율, 재시도 비율 | 코드 품질 근거 | expert_33 채점 결과 |
| 1-2 라벨 통일 후 재채점 | 첫 오류 줄 일치 (기준표 번호 기준) | 오세준 교수님 | 라벨 수정 + expert_33 채점 결과 |
| 1-3 A-05-B·A-07-B 재확인 | 두 답안이 여전히 모호한지 | 해결되면 질문 삭제 | 1-2 결과 |
| 1-4 설명문형 5건 변환 | 변환 전/후 판정 비교 | 오세준 교수님 라벨 확인 | 5건을 풀이 형태로 다시 쓴 답안 |
| 1-5 기권 손실 분류 | 시나리오 1·2 일치율 | 이용오 교수님 '코드 없음' 의도 | holdout 채점 결과 |
| 1-6 미완 답안 집계 | 미완 판정 수, 그중 R3가 붙은 수 | 미완 집계 방식 | holdout·expert 채점 결과 |
| 1-7 R6 사례 | R6 라벨 답안을 모델이 어떻게 판정했나 | R6 정의 범위 | holdout 채점 결과 |
| 1-8 손글씨 OCR 비교 | 엔진별 한글·수식 인식률, 교정 위험, 채점 변화 | OCR vs 타이핑 결정 | 손글씨 사진 + 정답 텍스트 |

미팅 때 꼭 숫자가 있어야 하는 건 **1-2, 1-5, 1-7, 1-8**이다. 1-8은 손글씨를 모으는 데 시간이 걸리니 가장 먼저 시작한다.

---

## 0. 준비 (처음 한 번)

```powershell
git checkout stabilize
pip install -r requirements.txt
pip install pytest
python -m pytest tests -q          # 전부 통과하는지 확인
```

Windows PowerShell에서는 한글 깨짐(cp949 오류)을 막기 위해 창마다 먼저 실행한다.

```powershell
$env:PYTHONUTF8 = "1"
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

채점 모델은 코드 기본값이 **Opus 5.5**다 (도구 호출 강제 끔, temperature 안 보냄). 다른 모델로 돌릴 때만 `$env:RV2_MODEL = "claude-haiku-4-5"`처럼 지정한다. 결과 파일에 모델·프롬프트·grader·코드북 버전이 모두 기록되고, 분석 보고서 첫 표에 나온다. **버전이 섞인 결과끼리는 비교하지 않는다.**

### 답안 파일

| 파일 | 만드는 법 | 저장소에 올리나 |
| --- | --- | --- |
| `answers/expert_33.csv` | `python experiments/export_expert33.py` (엑셀 04 시트에서 생성) | 올려도 됨 |
| `answers/converted_5.csv` | 위 명령이 틀을 같이 만든다. 사람이 채운다 (1-4) | 올려도 됨 |
| `answers/holdout45.csv` | 키트에서 복사 | **올리지 않는다** (holdout 봉인) |
| `answers/ocr_samples.csv` + 사진 | `experiments/templates/ocr_samples.csv`를 복사해서 채운다 (1-8) | 학생 손글씨라 **올리지 않는다** |

`export_expert33.py`는 이미 있는 파일을 덮어쓰지 않는다 (사람이 고친 라벨 보호). 새로 만들려면 `--force`.

---

## 1. 사람이 먼저 할 일

### 1-2용: 라벨을 기준표 번호로 통일
`answers/expert_33.csv`의 `첫_독립_오류_기준` 열을 고친다. 처음에는 `첫_독립_오류`(학생 줄 번호)가 그대로 복사돼 있다.

- 고칠 답안: **A-01-A, A-01-B, A-04-A, A-04-B** (학생 줄 번호 → 기준표 번호, 11/11 자료 C-1)
- 확인할 답안: **A-05-B, A-07-B** (모호. 1-3에서 결과 보고 판단)
- 기준표 번호는 엑셀 `06_LINE_JUDGMENT`의 `line_no`
- 원래 `첫_독립_오류` 열은 지우지 않는다 (예전 기준 비교용)

### 1-4용: 설명문형 5건을 풀이 형태로 변환
`answers/converted_5.csv`의 `학생_답안` 칸을 채운다. `원래_답안`에 원문이 있다.
대상: A-05-D, A-06-A, A-06-C, A-08-A, A-09-B. 형식은 11/11 자료의 A-06-C 예시처럼 `① … ② …`.
`학생_답안`이 빈 답안은 `run_batch.py`가 건너뛴다 (빈 답안에 비용 안 씀).

### 1-8용: 손글씨 표본 수집
- 고1 수준 손글씨 **30~50장** (지인 학생, 섭외 학교 한 반, 안 되면 팀원이 일부러 대충 쓴 것)
- 사진은 `answers/ocr/images/`에, 목록은 `answers/ocr/ocr_samples.csv`에 (틀: `experiments/templates/ocr_samples.csv`)
- `정답_텍스트`: 학생이 쓴 그대로 타이핑. **틀린 값도 그대로**, 줄은 줄바꿈으로 구분
- 일부 표본은 일부러 틀린 값을 쓰게 하고 그 식을 `교정검사` 열에 적는다 (예: `f(-2)=5`). 엔진이 이걸 맞는 값으로 고쳐버리는지 본다
- 한글 서술 줄, 수식만 있는 줄, 섞인 줄이 골고루 들어가게 한다

---

## 2. 채점 실행 (API 비용 발생)

```powershell
# expert_33 (1-1, 1-2, 1-3, 1-6): 33건 × 3회
python run_batch.py --answers answers/expert_33.csv --repeats 3

# holdout (1-5, 1-6, 1-7): 상태 '유효' 40건 × 3회
python run_batch.py --answers answers/holdout45.csv --repeats 3

# 변환 5건 (1-4): 5건 × 3회
python run_batch.py --answers answers/converted_5.csv --repeats 3
```

결과는 `results/stage3_<답안파일>_<모델>_<코드북>_<시각>.jsonl`로 쌓인다. 중간에 끊기면 그 파일을 버리고 다시 돌린다 (이어하기 기능은 없다).
비용 감: 11/11 자료 기준 Opus 5.5로 holdout 3회 약 $6.4. expert_33 3회도 비슷한 규모.

---

## 3. 분석 (비용 없음)

`--out`을 주면 보고서를 마크다운으로 저장한다. 결과 파일 경로는 실제 생성된 이름으로 바꾼다.

### expert_33 → 1-1, 1-2, 1-3, 1-6, 1-7
```powershell
python experiments/score_experiments.py `
  --results results/stage3_expert_33_....jsonl `
  --answers answers/expert_33.csv `
  --first-col 첫_독립_오류_기준 --first-basis criterion `
  --out experiments/out/expert33_report.md
```

예전 수치(첫 오류 줄 9/17, 학생 줄 번호 기준)와 비교하려면 같은 결과 파일로 기준만 바꿔 한 번 더 돌린다 (추가 비용 없음).
```powershell
python experiments/score_experiments.py --results results/stage3_expert_33_....jsonl `
  --answers answers/expert_33.csv --first-col 첫_독립_오류 --first-basis student --only 1-2
```

### holdout → 1-5, 1-6, 1-7
```powershell
python experiments/score_experiments.py `
  --results results/stage3_holdout45_....jsonl `
  --answers answers/holdout45.csv `
  --focus-ids "" --focus-code R6 `
  --out experiments/out/holdout_report.md
```
holdout CSV의 라벨 열 이름이 다르면 `--code-col`(주 코드), `--first-col`(첫 오류 줄), `--type-col`(오탐검사 표시)로 맞춘다. 없는 열을 주면 바로 알려 준다.
1-5 보고서 맨 아래 **기권 행 목록**이 이용오 교수님께 재문의할 때 넣을 답안 ID다.

### 변환 5건 → 1-4
```powershell
python experiments/compare_runs.py `
  --before results/stage3_expert_33_....jsonl --before-answers answers/expert_33.csv `
  --after  results/stage3_converted_5_....jsonl --after-answers answers/converted_5.csv `
  --out experiments/out/1-4_converted.md
```

### 손글씨 → 1-8
```powershell
$env:GEMINI_API_KEY = "..."
$env:MATHPIX_APP_ID = "..."; $env:MATHPIX_APP_KEY = "..."
$env:AZURE_VISION_ENDPOINT = "https://<이름>.cognitiveservices.azure.com"; $env:AZURE_VISION_KEY = "..."

python experiments/ocr_compare.py `
  --samples answers/ocr/ocr_samples.csv `
  --engines gemini,mathpix,claude,azure `
  --grade `
  --out experiments/out/1-8_ocr.md
```
- 키가 없는 엔진은 건너뛰고 표에 `키 없음`으로 남는다. 키가 있는 엔진부터 돌려도 된다
- `--grade`는 정답 텍스트와 OCR 결과를 각각 채점해 결과가 바뀌는지 본다. 표본 × (엔진 + 1)번 채점하므로 비용이 든다. 먼저 `--grade` 없이 인식률만 보고 붙여도 된다
- Claude 비전 모델은 `$env:OCR_CLAUDE_MODEL`로 바꿀 수 있다 (기본 Opus 5.5)

---

## 4. 미팅에 가져갈 것

| 보고서 | 볼 곳 | 가져갈 문장 예 |
| --- | --- | --- |
| expert33_report.md | 1-2 표 | "라벨을 기준표 번호로 통일한 뒤 첫 오류 줄 일치 9/17 → N/M" |
| expert33_report.md | 1-3 표 | A-05-B·A-07-B가 일치하면 질문 목록에서 삭제 |
| holdout_report.md | 1-5 표 | "'코드 없음' 의도에 따라 일치율 X/40 ~ Y/40. 기권 중 진짜 손실 K건" |
| holdout_report.md | 1-7 표 | "R6 라벨 N행 중 M행이 코드 없음으로 처리" |
| expert33·holdout 1-6 | 미완 표 | 미완 판정 수, R3가 붙은 행 (0이어야 정상) |
| 1-4_converted.md | 전체 | 변환 후 판정이 달라진 답안 수 + 교수님 라벨 확인 요청 |
| 1-8_ocr.md | 엔진 표 | "엔진별 한글/수식 CER, 교정 발생, 채점이 바뀐 비율 → OCR로 갈지 타이핑으로 갈지" |
| expert33_report.md | 1-1 표 | 형식 문제·재시도 비율 (코드 품질 근거) |

---

## 알아둘 점

- **1-1은 '강화 전' 기준선이 없다.** 프롬프트 출력 형식 규칙(1.3)을 측정 없이 먼저 적용하기로 했다. 지금 수치는 1.3 상태의 안정성이다. 강화 전과 비교가 꼭 필요하면 `backup-2026-11-24` 브랜치로 같은 답안을 돌리되, 그 버전에는 `extraction`·`first_issues` 기록이 없어 재시도·에러 비율만 비교할 수 있다.
- **기권의 정의**: 모델이 검수를 요청하고(review_needed) 주 코드를 비운 경우. 검증 문제로 검수에 간 경우는 기권이 아니다.
- **첫 오류 줄(모델)**: judgment가 wrong인 줄 중 번호가 가장 작은 줄. `--first-basis criterion`이면 그 줄의 기준표 번호, `student`면 학생 줄 번호와 비교한다.
- **1-8은 엔진 단독 비교다.** 여러 엔진을 합치는 대조 단계는 아직 없다. MyScript는 펜 획 좌표가 필요해서 사진으로는 비교할 수 없다 (태블릿 획 수집 화면이 생긴 뒤).
- **Azure 연동은 실제 키로 아직 확인하지 않았다.** 첫 실행 때 `1-8_ocr_raw.jsonl`에서 azure 행에 에러가 없는지 본다.
- 분석 스크립트 테스트: `python -m pytest tests/test_experiments.py -q`
