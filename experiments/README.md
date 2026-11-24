# 미팅 전 필수 실험 실행 안내

미팅 전에 숫자를 들고 가야 하는 실험 1-1 ~ 1-8을 돌리는 방법이다.

- **채점 정의는 키트와 같다.** 분석은 키트 채점기(`experiments/score_rv2.py`, 원본 `rv2_kit/score_rv2.py` 복사본)의 함수를 그대로 쓴다. 같은 결과 파일이면 키트와 같은 숫자가 나온다.
- 채점(API 호출)은 `run_batch.py`, 분석은 이 폴더의 스크립트. 분석은 LLM을 부르지 않는다 (비용 0).
- **키트에 있는 11/11 Opus 5.5 결과로 대부분의 실험을 새로 채점하지 않고 분석할 수 있다.**

## 한눈에

| 실험 | 무엇을 보나 | 근거로 쓰는 질문 | 새 채점 필요? | 상태 (11/24) |
| --- | --- | --- | --- | --- |
| 1-1 스키마 안정성 | 응답 형식이 깨지는 비율, 재시도 | 코드 품질 근거 | 아니오 (1.2 기준선). 1.3 수치는 재채점 시 | **완료** (1.2) |
| 1-2 라벨 통일 후 첫 오류 줄 | 기준표 번호 기준 일치율 | 오세준 교수님 | **아니오** — 라벨만 고치고 재분석 | 라벨 수정 대기 |
| 1-3 A-05-B·A-07-B | 두 답안이 여전히 모호한지 | 해결되면 질문 삭제 | 아니오 | **완료** |
| 1-4 설명문형 5건 변환 | 변환 전/후 판정 | 오세준 교수님 라벨 확인 | 예 (5건 × 3회) | 변환 답안 작성 대기 |
| 1-5 기권 손실 분류 | 시나리오별 일치율, 진짜 손실 | 이용오 교수님 '코드 없음' 의도 | 아니오 | **완료** |
| 1-6 미완 답안 | 미완 판정, R3 부착 | 미완 집계 방식 | 아니오 | **완료** |
| 1-7 R6 사례 | R6 라벨 답안의 AI 판정과 사유 | R6 정의 범위 | 아니오 | **완료** |
| 1-8 손글씨 OCR | 엔진별 인식률·교정 위험·채점 변화 | OCR vs 타이핑 | 예 (OCR API) | 손글씨 수집 대기 |

---

## 0. 준비 (처음 한 번)

```powershell
git checkout stabilize
pip install -r requirements.txt
pip install pytest
python -m pytest tests -q
```

PowerShell 창마다 먼저 (한글 깨짐 cp949 오류 방지):
```powershell
$env:PYTHONUTF8 = "1"
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # 채점할 때만 필요
```

답안 파일 준비 — 키트 파일을 `answers/`로 복사한다. 이미 있으면 덮어쓰지 않는다 (새로 만들려면 `--force`).
```powershell
python experiments/prepare_answers.py --kit "C:\Users\USER\Desktop\연구\rv2_kit"
```

| 만들어지는 파일 | 내용 | 저장소에 올리나 |
| --- | --- | --- |
| `answers/expert_33.csv` | 키트 복사본 + `전문가_첫오류_통일` 열 (1-2용, 처음엔 원래 라벨 복사) | 올려도 됨 |
| `answers/holdout45.csv` | 키트 복사본 | **안 올라감** (.gitignore, holdout 봉인) |
| `answers/converted_5.csv` | 1-4용 틀. `원래_답안`에 원문, `학생_답안`은 빈칸 | 올려도 됨 |

키트 원본(`rv2_kit/answers/`)은 건드리지 않는다. 라벨 수정은 `answers/` 사본에서만 한다.

---

## 1. 새 채점 없이 바로 하는 분석 (11/11 Opus 5.5 결과)

결과 파일: `rv2_kit/results/stage3_expert_33_opus55_20261111_1422.jsonl`, `stage3_holdout45_opus55_20261111_1531.jsonl`
(모델 Opus 5.5, 프롬프트 hongsu-rv2-1.2, 코드북 R-v3, 반복 3회)

```powershell
$KIT = "C:\Users\USER\Desktop\연구\rv2_kit"
python experiments/score_experiments.py `
  --results "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" `
  --answers answers/expert_33.csv --out experiments/out/expert33_report.md

python experiments/score_experiments.py `
  --results "$KIT\results\stage3_holdout45_opus55_20261111_1531.jsonl" `
  --answers answers/holdout45.csv --focus-ids "" --out experiments/out/holdout_report.md
```

### 1-2: 라벨 통일 → 재분석 (비용 0)
1. `answers/expert_33.csv`의 `전문가_첫오류_통일` 열에서 아래 답안을 **기준표 번호**(엑셀 `06_LINE_JUDGMENT`의 `line_no`)로 고친다.
   - 고칠 답안: **A-01-A, A-01-B, A-04-A, A-04-B** (11/11 자료 C-1)
   - 확인할 답안: **A-05-B, A-07-B**
   - `①` 같은 원문자나 숫자 모두 된다. 원래 `전문가_첫오류` 열은 그대로 둔다
2. 같은 결과 파일로 다시 분석한다.
```powershell
python experiments/score_experiments.py `
  --results "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" `
  --answers answers/expert_33.csv --first-col 전문가_첫오류_통일 --only 1-2,1-3 `
  --out experiments/out/1-2_unified.md
```
라벨은 사람이 기준표를 보고 정한다. 보고서 표의 'AI 첫 줄'을 보고 라벨을 맞추면 안 된다 (평가 데이터 오염).

---

## 2. 새 채점이 필요한 실험 (API 비용)

| 실행 | 명령 | 규모 |
| --- | --- | --- |
| 1-4 변환 5건 | `python run_batch.py --answers answers/converted_5.csv --repeats 3` | 15회 호출 |
| (선택) 1.3 코드로 expert 재채점 | `python run_batch.py --answers answers/expert_33.csv --repeats 3` | 99회 |
| (선택) 1.3 코드로 holdout 재채점 | `python run_batch.py --answers answers/holdout45.csv --repeats 3` | 120회 |

- 1-4: `answers/converted_5.csv`의 `학생_답안`을 먼저 채운다 (`① … ② …` 풀이 형태, 11/11 자료 A-06-C 예시처럼). 빈 답안은 `run_batch.py`가 건너뛴다.
- 1.3 재채점은 안정화 작업(형식 규칙 추가, 재시도 조건 확대) 뒤에도 수치가 그대로인지 확인하는 용도다. 결과 파일에 `grader_version=rv2-grader-1.3`이 찍히고, 1-1의 '첫 호출 사유'까지 나온다.
- 비용 감: 11/11 자료 기준 Opus 5.5로 holdout 3회 약 $6.4.

1-4 분석:
```powershell
python experiments/compare_runs.py `
  --before "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" --before-answers answers/expert_33.csv `
  --after  results/stage3_converted_5_....jsonl --after-answers answers/converted_5.csv `
  --out experiments/out/1-4_converted.md
```

### 1-8: 손글씨 OCR 비교
준비: 고1 수준 손글씨 **30~50장**. 사진은 `answers/ocr/images/`, 목록은 `answers/ocr/ocr_samples.csv` (틀: `experiments/templates/ocr_samples.csv`). `answers/ocr/`는 저장소에 안 올라간다.
- `정답_텍스트`: 학생이 쓴 그대로, **틀린 값도 그대로**, 줄은 줄바꿈으로
- 일부는 일부러 틀린 값을 쓰게 하고 그 식을 `교정검사` 열에 (예: `f(-2)=5`)
- 한글 서술 줄 / 수식만 / 섞인 줄이 골고루 들어가게

```powershell
$env:GEMINI_API_KEY = "..."
$env:MATHPIX_APP_ID = "..."; $env:MATHPIX_APP_KEY = "..."
$env:AZURE_VISION_ENDPOINT = "https://<이름>.cognitiveservices.azure.com"; $env:AZURE_VISION_KEY = "..."

python experiments/ocr_compare.py --samples answers/ocr/ocr_samples.csv `
  --engines gemini,mathpix,claude,azure --out experiments/out/1-8_ocr.md          # 인식률만 (OCR 비용)
python experiments/ocr_compare.py --samples answers/ocr/ocr_samples.csv `
  --engines gemini,mathpix,claude,azure --grade --out experiments/out/1-8_ocr.md  # + 채점 영향 (채점 비용 추가)
```
키가 없는 엔진은 건너뛰고 `키 없음`으로 표시된다. Claude 비전 모델은 `$env:OCR_CLAUDE_MODEL`로 바꿀 수 있다 (기본 Opus 5.5).

---

## 3. 보고서 읽는 법

| 섹션 | 볼 것 |
| --- | --- |
| 1-1 | 도구/텍스트/없음 호출 수, 재시도·에러 비율. 1.3 결과면 첫 호출 사유(형식·검증)도 |
| 1-2 | 첫 오류 줄 일치 (키트 정의: 1회차 기준) + 회차별 + 답안별 표 |
| 1-3 | 지정 답안의 회차별 판정·다수결·코드 일치 |
| 1-5 | 키트 기준 / 기권 인정 / 기권+코드누락 인정 세 줄과 '진짜 손실' 답안 목록 (모델 사유 포함) |
| 1-6 | 라벨 미완 답안의 AI 판단, AI가 미완으로 본 답안, 미완인데 R3가 붙은 회차 |
| 1-7 | 지정 코드(기본 R6) 라벨 답안의 AI 판정과 코드를 비운 사유, 라벨×AI 교차표 |

키트 정의 요약 (`score_rv2.py`):
- 답안별 주 코드 = 반복 다수결. 기권(`review_needed`)·코드누락(wrong 줄은 있는데 코드 없음)은 표에서 빼고, 남는 표가 없으면 그대로 기권/코드누락
- 라벨이 '코드 없음'이면 AI가 오류없음·미완일 때만 일치
- 첫 오류 줄 = 첫 wrong 줄의 `matched_criterion` (기준표 번호), 1회차 결과
- holdout: 상태 '유효' 40건, H-09·H-13 라벨은 R-v3에 따라 R2로 집계 (`--no-v3-relabel`로 끔)

---

## 알아둘 점

- **분석 정의를 바꾸지 말 것.** `experiments/score_rv2.py`는 키트 원본 복사본이다. 정의를 바꾸려면 키트 원본과 같이 바꾸고 교수님 보고 수치와의 차이를 남긴다.
- **1-1의 '강화 전' 기준선** = 11/11 결과 (프롬프트 1.2). 1.3 재채점을 하면 전후 비교가 된다. 1.2 결과에는 `first_issues` 기록이 없어 결과 출처는 응답 원문 블록으로 셌다.
- **response_complete와 '미완' 담론은 다른 개념일 수 있다.** 코드북 R-v3의 `response_complete`는 '요구한 기준을 다 다뤘는가'(missing_criteria)이고, 담론 2번의 미완은 '중간에 멈췄는가'다. 1-6 표에서 코드가 붙은 답안도 response_complete=false인 경우가 있다. 미완 처리 규칙을 만들기 전에 이 차이를 정리해야 한다.
- **Azure 연동은 실제 키로 아직 확인하지 않았다.** 첫 실행 때 `1-8_ocr_raw.jsonl`의 azure 행을 확인한다.
- **1-8은 엔진 단독 비교다.** 여러 엔진을 합치는 대조 단계는 아직 없다. MyScript는 펜 획 좌표가 필요해서 사진으로는 비교할 수 없다.
- 테스트: `python -m pytest tests/test_experiments.py -q`
