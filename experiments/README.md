# 실험 실행 안내 — 교수님 점검 전에 먼저 하는 실험

교수님 점검(다음 미팅) 없이 우리끼리 먼저 하기로 한 실험을 정리했다. 두 종류다.

- **먼저 바꾼 것 검증**: 교수님 확인 없이 먼저 적용한 코드 변경(hongsu 1.3 안정화, 채점 모델 Opus 5.5)이 결과를 바꾸지 않았는지 → 1-1
- **미팅에 가져갈 숫자**: 교수님께 여쭐 질문의 근거 → 1-2 ~ 1-8

미팅에서 결정을 받은 뒤에 하는 실험(확신도 기준선, 기권 차단 프롬프트, 캐스케이드, 주 코드 선정 규칙)은 여기 없다.

채점(API 호출)은 `run_batch.py`, 분석은 이 폴더의 스크립트로 한다. 분석 스크립트는 LLM을 부르지 않는다 (비용 0).
일치율 정의는 팀 채점기 `experiments/score_rv2.py`(키트 `rv2_kit/score_rv2.py` 복사본)를 그대로 쓴다.

---

## 기준 결과 파일 — 11/10·11/11 최신본만 쓴다

키트 `rv2_kit/results/` 안에서 아래 파일만 기준으로 삼는다.

| 파일 | 모델 | 답안 | 반복 | 쓰는 곳 |
| --- | --- | --- | --- | --- |
| `stage3_expert_33_opus55_20261111_1422.jsonl` | Opus 5.5 | expert 33 | 3 | **1-1a, 1-2, 1-3, 1-4(변환 전), 1-6** |
| `stage3_holdout45_opus55_20261111_1531.jsonl` | Opus 5.5 | holdout 40 | 3 | **1-1a, 1-5, 1-6, 1-7** |
| `stage3_expert_33_server_server_20261111_0859_haiku45.jsonl` | Haiku 4.5 | expert 33 | 3 | 모델 비교용 (선택) |
| `stage3_expert_33_server_server_20261111_0932_haiku55.jsonl` | Haiku 5.5 | expert 33 | 3 | 모델 비교용 (선택) |
| `stage3_expert_33_server_server_20261111_1256_opus46.jsonl` | Opus 4.6 | expert 33 | 3 | 모델 비교용 (선택) |
| `stage3_holdout45_server_server_20261110_1242.jsonl` | Haiku 4.5 | holdout 40 | 3 | 모델 비교용 (선택) |
| `stage3_holdout45_server_server_20261111_1316_opus46.jsonl` | Opus 4.6 | holdout 40 | 3 | 모델 비교용 (선택) |
| `stage3_holdout45_server_server_20261111_1844_haiku55.jsonl` | Haiku 5.5 | holdout 40 | 3 | 모델 비교용 (선택, 프롬프트 1.2.1) |
| `stage3_holdout45_server_server_20261111_1909_haiku55_v12.jsonl` | Haiku 5.5 | holdout 40 | 3 | 모델 비교용 (선택) |

쓰지 않는 파일: 10/27 결과, 11/10 00:46~01:00 세 파일(키트 단독 실행, 33건 × 1회, 코드북 버전 기록 없음), 이름이 `.jsonl.jsonl`로 끝나는 파일(같은 내용의 중복본).
모든 기준 파일은 프롬프트 hongsu-rv2-1.2, 코드북 R-v3이다. 1.3 코드로 새로 채점한 결과와 섞지 않는다.

---

## 실험 한눈에

| 실험 | 확인할 것 | 결과 파일 | 라벨 파일 | 사람이 먼저 할 일 | 새 채점 | 상태 (11/24) |
| --- | --- | --- | --- | --- | --- | --- |
| 1-1a 스키마 안정성 (1.2 기준선) | 형식 깨짐·재시도·에러 비율 | opus55 expert·holdout | expert_33, holdout45 | 없음 | 아니오 | **완료** |
| 1-1b 1.3 재채점 검증 | 1.3에서도 형식이 안 깨지는지, 판정이 1.2와 같은지 | 새로 만듦 | expert_33, holdout45 | 없음 | **예** (expert 99 + holdout 120 호출) | 실행 대기 |
| 1-2 라벨 통일 후 첫 오류 줄 | 기준표 번호로 통일한 라벨 기준 일치율 | opus55 expert | expert_33 (통일 열) | **라벨 4건 수정** | 아니오 | 라벨 수정 대기 |
| 1-3 A-05-B·A-07-B | 두 답안이 여전히 모호한지 | opus55 expert | expert_33 | 없음 (1-2 뒤 한 번 더) | 아니오 | **완료** (1차) |
| 1-4 설명문형 5건 변환 | 풀이 형태로 바꾸면 판정이 바뀌는지 | opus55 expert + 새로 만듦 | expert_33, converted_5 | **5건 다시 쓰기** | **예** (15 호출) | 작성 대기 |
| 1-5 기권 손실 분류 | '코드 없음' 해석별 일치율, 진짜 손실 | opus55 holdout | holdout45 | 없음 | 아니오 | **완료** |
| 1-6 미완 답안 | 미완 판정, 미완인데 R3가 붙은 경우 | opus55 expert·holdout | expert_33, holdout45 | 없음 | 아니오 | **완료** |
| 1-7 R6 사례 | R6 라벨 답안의 AI 판정과 사유 | opus55 holdout | holdout45 | 없음 | 아니오 | **완료** |
| 1-8 손글씨 OCR | 엔진별 한글·수식 인식률, 교정 위험, 채점 변화 | 새로 만듦 | ocr_samples | **손글씨 30~50장 + 정답 타이핑** | **예** (OCR API) | 수집 대기 |

완료된 보고서: `rv2_kit/results/experiments_20261124/` (expert33·holdout45 각 1개)

---

## 0. 준비 (처음 한 번)

```powershell
git checkout stabilize
pip install -r requirements.txt
pip install pytest
python -m pytest tests -q

$env:PYTHONUTF8 = "1"                              # PowerShell 창마다 (한글 cp949 오류 방지)
$env:ANTHROPIC_API_KEY = "sk-ant-..."             # 새 채점할 때만
$KIT = "C:\Users\USER\Desktop\연구\rv2_kit"         # 아래 명령에서 씀

python experiments/prepare_answers.py --kit $KIT
```

`prepare_answers.py`가 만드는 파일 (이미 있으면 덮어쓰지 않음, 새로 만들려면 `--force`):

| 파일 | 내용 | 저장소 |
| --- | --- | --- |
| `answers/expert_33.csv` | 키트 복사본 + `전문가_첫오류_통일` 열 (1-2용, 처음엔 원래 라벨 복사) | 올려도 됨 |
| `answers/holdout45.csv` | 키트 복사본 | 안 올라감 (.gitignore) |
| `answers/converted_5.csv` | 1-4용 틀 (`원래_답안`에 원문, `학생_답안` 빈칸) | 올려도 됨 |

키트 원본 파일은 건드리지 않는다. 라벨 수정은 `answers/` 사본에서만 한다.

---

## 1-1 스키마 안정성 / 먼저 바꾼 코드 검증

### 1-1a 1.2 기준선 — 완료, 비용 0
기준 결과 파일의 응답 원문에서 셌다. 결과: Opus 5.5(도구 강제 끔) 219번 호출 모두 도구 응답, 재시도·에러·남은 검증 문제 0건.
```powershell
python experiments/score_experiments.py --results "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" `
  --answers answers/expert_33.csv --only 1-1
python experiments/score_experiments.py --results "$KIT\results\stage3_holdout45_opus55_20261111_1531.jsonl" `
  --answers answers/holdout45.csv --only 1-1
```

### 1-1b 1.3 코드로 재채점 — 새 채점 필요
목적: 교수님 확인 없이 먼저 적용한 1.3 변경(출력 형식 규칙 추가, 재시도 조건 확대)이 판정을 바꾸지 않았는지 확인. 미팅 숫자를 1.3 기준으로 바꿔 말할 수 있는지가 여기서 정해진다.

1) 채점 (현재 코드 = 모델 Opus 5.5, grader 1.3):
```powershell
python run_batch.py --answers answers/expert_33.csv --repeats 3
python run_batch.py --answers answers/holdout45.csv --repeats 3
```
→ `results/stage3_expert_33_claude-opus-5-5_R-v3_<시각>.jsonl`, `results/stage3_holdout45_claude-opus-5-5_R-v3_<시각>.jsonl`

2) 1.3 안정성 (첫 호출 형식 문제·재시도 사유까지 나온다):
```powershell
python experiments/score_experiments.py --results results/stage3_expert_33_claude-opus-5-5_R-v3_<시각>.jsonl `
  --answers answers/expert_33.csv --out experiments/out/1-1b_expert33_report.md
python experiments/score_experiments.py --results results/stage3_holdout45_claude-opus-5-5_R-v3_<시각>.jsonl `
  --answers answers/holdout45.csv --focus-ids "" --out experiments/out/1-1b_holdout45_report.md
```

3) 1.2(11/11)와 판정 비교:
```powershell
python experiments/compare_runs.py --ids all --title "1-1b expert 1.2 vs 1.3" --before-tag 1.2 --after-tag 1.3 `
  --before "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" --before-answers answers/expert_33.csv `
  --after results/stage3_expert_33_claude-opus-5-5_R-v3_<시각>.jsonl --after-answers answers/expert_33.csv `
  --out experiments/out/1-1b_expert33_compare.md
python experiments/compare_runs.py --ids all --title "1-1b holdout 1.2 vs 1.3" --before-tag 1.2 --after-tag 1.3 `
  --before "$KIT\results\stage3_holdout45_opus55_20261111_1531.jsonl" --before-answers answers/holdout45.csv `
  --after results/stage3_holdout45_claude-opus-5-5_R-v3_<시각>.jsonl --after-answers answers/holdout45.csv `
  --out experiments/out/1-1b_holdout45_compare.md
```
볼 것: 비교 보고서 맨 아래 "다수결 판정이 달라진 답안" 수. 0~1건이면 1.3 변경이 판정에 영향 없다고 말할 수 있다. 여러 건이면 달라진 답안의 모델 사유를 보고 형식 규칙 탓인지 확인한다.
비용 감: 11/11 자료 기준 Opus 5.5 holdout 3회 약 $6.4, expert 3회도 비슷한 규모.

---

## 1-2 라벨 통일 후 첫 오류 줄 — 비용 0, 라벨 수정 필요

현재(라벨 그대로): **9/17** (1회차 기준).
1) `answers/expert_33.csv`의 `전문가_첫오류_통일` 열에서 아래를 **기준표 번호**(엑셀 `06_LINE_JUDGMENT`의 `line_no`)로 고친다.
   - 고칠 답안: **A-01-A, A-01-B, A-04-A, A-04-B** (11/11 자료 C-1)
   - 확인할 답안: **A-05-B, A-07-B**
   - `①`이든 `1`이든 된다. 원래 `전문가_첫오류` 열은 그대로 둔다
2) 같은 결과 파일로 다시 분석:
```powershell
python experiments/score_experiments.py --results "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" `
  --answers answers/expert_33.csv --first-col 전문가_첫오류_통일 --only 1-2,1-3 `
  --out experiments/out/1-2_unified.md
```
라벨은 기준표를 보고 사람이 정한다. 보고서의 'AI 첫 줄'을 보고 라벨을 맞추면 평가 데이터가 오염된다.
1-1b를 돌렸다면 같은 명령에서 `--results`만 1.3 결과로 바꿔 한 번 더 본다.

## 1-3 A-05-B·A-07-B — 완료 (1-2 뒤 한 번 더)

결과: 두 답안 모두 코드는 3회 모두 맞음(R6, R2). 첫 오류 줄만 라벨과 다름(1 vs 3, 3 vs 4) → 라벨 번호 문제로 보인다. 1-2 명령의 출력에 1-3 표도 같이 나온다.

## 1-4 설명문형 5건 변환 — 새 채점 15호출

1) `answers/converted_5.csv`의 `학생_답안` 5칸을 풀이 형태(`① … ② …`, 11/11 자료의 A-06-C 예시처럼)로 채운다. 대상: A-05-D, A-06-A, A-06-C, A-08-A, A-09-B. 빈칸인 답안은 `run_batch.py`가 건너뛴다.
2) 채점과 비교:
```powershell
python run_batch.py --answers answers/converted_5.csv --repeats 3
python experiments/compare_runs.py `
  --before "$KIT\results\stage3_expert_33_opus55_20261111_1422.jsonl" --before-answers answers/expert_33.csv `
  --after results/stage3_converted_5_claude-opus-5-5_R-v3_<시각>.jsonl --after-answers answers/converted_5.csv `
  --out experiments/out/1-4_converted.md
```
주의: 변환 전은 1.2(11/11), 변환 후는 1.3 코드로 채점된다. 1-1b에서 1.2와 1.3 판정이 같다고 확인되면 그대로 비교하고, 아니면 1-1b의 1.3 expert 결과를 `--before`로 쓴다.

## 1-5 기권 손실 분류 — 완료, 비용 0

결과 (holdout 40건, 답안 단위 다수결):

| 기준 | 일치 |
| --- | --- |
| 키트 기준 (기권·코드누락 = 불일치) | 31/40 |
| '코드 없음' 라벨에 대한 기권을 일치로 인정 | 35/40 |
| 기권 + 코드누락까지 인정 | 36/40 (11/11 자료의 36/40) |

진짜 손실(라벨에 코드가 있는데 기권)은 **H-15 1건**. 나머지 기권은 모두 '코드 없음' 라벨 답안이다. 보고서의 기권 목록이 이용오 교수님 재문의에 넣을 답안 ID다.
```powershell
python experiments/score_experiments.py --results "$KIT\results\stage3_holdout45_opus55_20261111_1531.jsonl" `
  --answers answers/holdout45.csv --focus-ids "" --only 1-5
```

## 1-6 미완 답안 — 완료, 비용 0

결과: 라벨이 미완인 답안(holdout H-37·H-43, expert A-08-A·A-08-D)은 AI도 3회 모두 미완으로 봄. 미완인데 R3가 붙은 경우 0건.
다만 holdout에서 코드가 붙은 답안 5건(H-12, H-18, H-25, H-30, H-45)도 AI가 한 번 이상 `response_complete=false`로 표시했다. 코드북의 `response_complete`는 '요구 기준을 다 다뤘나'이고 담론 2번의 미완은 '중간에 멈췄나'라서, 미완 처리 규칙을 정하기 전에 이 차이를 정리해야 한다.
```powershell
python experiments/score_experiments.py --results "$KIT\results\stage3_holdout45_opus55_20261111_1531.jsonl" `
  --answers answers/holdout45.csv --focus-ids "" --only 1-6
```

## 1-7 R6 사례 — 완료, 비용 0

결과: R6 라벨 3건 중 H-15만 기권. 모델 사유: "R6은 y에 대해 정리하는 과정의 이항·부호 오류로 정의돼 있는데, 이 학생은 정리는 옳게 하고 상수항 부호를 잘못 읽었다" → R6 정의 범위 질문의 근거.
```powershell
python experiments/score_experiments.py --results "$KIT\results\stage3_holdout45_opus55_20261111_1531.jsonl" `
  --answers answers/holdout45.csv --focus-ids "" --only 1-7 --focus-code R6
```

## 1-8 손글씨 OCR — 수집 필요, OCR API 비용

1) 손글씨 **30~50장** (고1 수준. 지인 학생, 섭외 학교 한 반, 안 되면 팀원이 일부러 대충 쓴 것)
   - 사진: `answers/ocr/images/`, 목록: `answers/ocr/ocr_samples.csv` (틀: `experiments/templates/ocr_samples.csv`). `answers/ocr/`는 저장소에 안 올라간다
   - `정답_텍스트`: 학생이 쓴 그대로, **틀린 값도 그대로**, 줄은 줄바꿈으로
   - 일부는 일부러 틀린 값을 쓰게 하고 그 식을 `교정검사` 열에 (예: `f(-2)=5`)
   - 한글 서술 줄 / 수식만 / 섞인 줄이 골고루
2) 실행:
```powershell
$env:GEMINI_API_KEY = "..."
$env:MATHPIX_APP_ID = "..."; $env:MATHPIX_APP_KEY = "..."
$env:AZURE_VISION_ENDPOINT = "https://<이름>.cognitiveservices.azure.com"; $env:AZURE_VISION_KEY = "..."

python experiments/ocr_compare.py --samples answers/ocr/ocr_samples.csv `
  --engines gemini,mathpix,claude,azure --out experiments/out/1-8_ocr.md            # 인식률만
python experiments/ocr_compare.py --samples answers/ocr/ocr_samples.csv `
  --engines gemini,mathpix,claude,azure --grade --out experiments/out/1-8_ocr.md    # + 채점 영향 (채점 비용 추가)
```
키가 없는 엔진은 건너뛰고 `키 없음`으로 표시된다. Claude 비전 모델은 `$env:OCR_CLAUDE_MODEL`로 바꿀 수 있다 (기본 Opus 5.5).

---

## 순서 제안

1. 1-8 손글씨 수집 시작 (가장 오래 걸림)
2. 1-2 라벨 4건 수정 → 재분석 (비용 0)
3. 1-1b 재채점 (expert·holdout) → 1.2와 판정 비교
4. 1-4 변환 5건 작성 → 채점·비교
5. 1-8 OCR 비교

---

## 보고서 읽는 법 / 정의

| 섹션 | 볼 것 |
| --- | --- |
| 1-1 | 도구/텍스트/없음 호출 수, 재시도·에러 비율. 1.3 결과면 첫 호출 사유(형식·검증)도 |
| 1-2 | 첫 오류 줄 일치 (1회차 기준) + 회차별 + 답안별 표 |
| 1-3 | 지정 답안의 회차별 판정·다수결·코드 일치 |
| 1-5 | 세 기준의 일치율, 기권·코드누락 답안 목록 (모델 사유 포함) |
| 1-6 | 라벨 미완 답안의 AI 판단, AI가 미완으로 본 답안, 미완인데 R3가 붙은 회차 |
| 1-7 | 지정 코드 라벨 답안의 AI 판정과 사유, 라벨×AI 교차표 |
| compare_runs | 답안별 전/후 판정, 다수결이 달라진 답안 수 |

팀 채점기(`score_rv2.py`) 정의:
- 답안별 주 코드 = 반복 다수결. 기권(`review_needed`)·코드누락(wrong 줄은 있는데 코드 없음)은 표에서 빼고, 남는 표가 없으면 그대로 기권/코드누락
- 라벨이 '코드 없음'이면 AI가 오류없음·미완일 때만 일치
- 첫 오류 줄 = 첫 wrong 줄의 `matched_criterion` (기준표 번호), 1회차 결과
- holdout: 상태 '유효' 40건, H-09·H-13 라벨은 R-v3에 따라 R2로 집계 (`--no-v3-relabel`로 끔)

## 알아둘 점

- `experiments/score_rv2.py`는 키트 원본 복사본이다. 정의를 바꾸려면 키트 원본과 같이 바꾼다. 교수님께 드린 자료는 채점 결과·이유를 정리한 엑셀이고 이 스크립트 출력과는 별개다.
- 결과 파일마다 모델·프롬프트·grader·코드북 버전이 보고서 첫 표에 나온다. **버전이 섞인 결과끼리는 같은 표에서 비교하지 않는다.**
- 1.2 결과에는 `first_issues` 기록이 없어, 1-1a의 결과 출처는 응답 원문 블록 종류로 셌다.
- Azure 연동은 실제 키로 아직 확인하지 않았다. 첫 실행 때 `1-8_ocr_raw.jsonl`의 azure 행을 본다.
- 1-8은 엔진 단독 비교다. 여러 엔진을 합치는 대조 단계는 아직 없다. MyScript는 펜 획 좌표가 필요해서 사진으로는 비교할 수 없다.
- 테스트: `python -m pytest tests/test_experiments.py -q`
