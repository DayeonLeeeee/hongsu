"""
Opus 채점 프롬프트 (감점형 100점, 4기준 × 25점).

빌드 함수:
  build(grading_mode, problem_text, model_answer, ...,
        observation_points, deduction_scale, steps_str, solution_text)
  build_retry(base_prompt, failed_list, ocr_base)
"""

VERSION = "grading-v2.3-2026-09-01"

SYSTEM = ("당신은 고등학교 1학년 「함수와 그래프」 단원의 서논술형 답안을 "
          "감점형 루브릭으로 정확하게 채점하는 수학 교사입니다. "
          "학생 풀이 원문의 증거에 기반해서만 감점하며, 절대 지어내지 않습니다.")

CRITERIA = ["개념 사용", "근거 제시", "계산·표현", "결론"]


_ROLE_CONTEXT = """[역할과 맥락]
당신은 고등학교 1학년 수학 교사이며, 「함수와 그래프」 단원의 서논술형 답안을 채점합니다.
학생은 15~16세, 태블릿에 손글씨로 풀이를 씁니다.
채점 결과는 학생에게 바로 화면으로 전달되므로 정확성과 근거가 특히 중요합니다.

[채점 철학 — 감점형]
학생은 각 기준 25점에서 시작합니다. 잘못한 것을 근거와 함께 지적하며 감점합니다.
- 잘못이 없으면 만점 (25점).
- 감점 사유는 반드시 학생 풀이 원문에서 인용할 수 있어야 합니다.
- 인용할 수 없는 감점은 감점하지 마세요."""


_GOOD_EXAMPLE = """[좋은 채점 예시 — 감점 근거 명시]

문제: f(x)=2x-1, g(x)=x²+3일 때 (f∘g)(-2)를 구하시오. (정답: 13)

학생 풀이:
  Step 1: f∘g는 f를 먼저 적용한다
  Step 2: f(-2) = 2×(-2)-1 = -5
  Step 3: g(-5) = (-5)²+3 = 28
  Step 4: (f∘g)(-2) = 28

채점 결과 (감점형):
  steps:
    Step 1: wrong    — "f∘g는 f를 먼저" (합성함수 순서 오류)
    Step 2: depends  — depends_on_line_id=1
    Step 3: depends  — depends_on_line_id=1
    Step 4: depends  — depends_on_line_id=1

  rubric_scores:
    - 개념 사용: 25 - 15 = 10
      deductions: [{points:-15, reason:"합성함수 정의 오해 (안쪽부터 계산 원칙 미준수)",
                    line_ref:1, evidence:"f∘g는 f를 먼저 적용한다"}]
    - 근거 제시: 25 - 5 = 20
      deductions: [{points:-5, reason:"순서를 뒤바꾼 근거 서술 없음",
                    line_ref:1, evidence:"f∘g는 f를 먼저 적용한다"}]
    - 계산·표현: 25 (감점 없음, 계산 자체는 정확)
      deductions: []
    - 결론: 25 - 20 = 5
      deductions: [{points:-20, reason:"최종 답 오답 (정답 13, 응답 28)",
                    line_ref:4, evidence:"(f∘g)(-2) = 28"}]

  total_score: 60/100
  is_correct: false

주목:
- 앞 오류에서 파생된 계산은 depends로 표시하고 중복 감점 안 함
- 계산·표현은 실제로 (-5)²=25 계산은 맞으므로 만점
- 각 감점마다 evidence 인용, line_ref 명시"""


_BAD_EXAMPLE = """[나쁜 채점 예시 — 이런 식으로 하면 안 됨]

× 감점 사유 뭉뚱그림:
  deductions: [{points:-25, reason:"전반적으로 부족함", line_ref:0, evidence:""}]
  → 어느 줄, 무엇이 부족한지 없음. 학생이 뭘 고칠지 모름.

× 한 번에 다 깎기:
  개념 사용: 25 - 25 = 0  (사유 1개로 25점 몽땅)
  → 여러 실수가 있으면 각각 근거와 함께 나눠 감점.

× 없는 증거 지어내기:
  evidence: "학생이 잘 몰라서 계산이 틀림"
  → evidence는 학생 풀이 원문에서 문자 그대로. 서술 금지.

× 앞 오류 파생을 이중 감점:
  Step 1 개념 오류로 -15, Step 2/3/4에서 각각 계산 오류라며 또 감점
  → depends 단계는 중복 감점 금지.

× 정답 학생한테 "더 좋은 방법" 감점:
  정답인데 "다른 풀이가 더 낫다"며 -5
  → 정답이면 감점 없음. 비역행 원칙."""


def _build_scale_section(deduction_scale: dict) -> str:
    """감점 척도표 섹션 (교수님 제공 데이터, 없으면 빈 문자열)"""
    if not deduction_scale:
        return ""
    lines = []
    for criterion in CRITERIA:
        if criterion in deduction_scale:
            lines.append(f"- {criterion} (25점): {deduction_scale[criterion]}")
    if not lines:
        return ""
    return "\n[감점 척도 — 이 문항에 적용 (교수 제공)]\n" + "\n".join(lines) + "\n"


def _build_obs_section(observation_points: str) -> str:
    if not observation_points:
        return ""
    return f"\n[관찰 지점 — 이 문제에서 특히 주의해서 볼 부분]\n{observation_points}\n"


def build(
    grading_mode: str,       # 'image' | 'text'
    problem_text: str,
    model_answer: str,
    required_reasoning: str,
    correct_answer: str,
    observation_points: str,
    deduction_scale: dict,
    steps_str: str,          # OCR 단계 요약 (image mode)
    solution_text: str,      # 학생 답안 텍스트 (text mode)
) -> str:
    """감점형 채점 프롬프트 조립"""

    if grading_mode == "image":
        prompt_intro = "첨부된 이미지는 학생의 손글씨 풀이입니다. 아래 문제 정보와 대조하여 채점하세요."
        solution_section = f"[학생 풀이 OCR (참고, 이미지가 항상 우선)]\n{steps_str}"
    else:
        prompt_intro = "아래 학생 풀이 텍스트를 문제 정보와 대조하여 채점하세요."
        solution_section = f"[학생 풀이]\n{solution_text}"

    obs_section = _build_obs_section(observation_points)
    scale_section = _build_scale_section(deduction_scale)

    return f"""{_ROLE_CONTEXT}

{prompt_intro}

[문제 텍스트]
{problem_text if problem_text else "(제공 안 됨)"}

[모범답안]
{model_answer if model_answer else "(없음)"}

[필수 풀이 근거]
{required_reasoning if required_reasoning else "(없음)"}

[알려진 정답]
{correct_answer if correct_answer else "(없음, 스스로 풀어 확인)"}
{obs_section}
{solution_section}

[채점 절차 — 순서대로]
1. 문제를 직접 풀어 정답과 필요 근거를 확정합니다.
2. 학생 풀이를 첫 줄부터 순서대로 읽으며 각 줄의 status를 판정합니다.
3. 4기준별로 감점 사유를 근거와 함께 기록합니다.
4. total_score 및 is_correct를 산정합니다.

[줄별 status 6종]
- ok: 논리·계산 모두 올바름.
- wrong: 독립적 오류 (앞과 무관하게 새로 발생). 다른 방법을 써도 논리적으로 타당하면 ok, 실제 수학적 오류만 wrong.
- depends: 앞 줄 오류에서 전파된 결과. `depends_on_line_id`에 원인 줄 번호 명시. 앞 오류만 없으면 성립하는 경우.
- insufficient: 결과는 맞지만 근거 서술이 부족함 (전제·조건 확인 생략).
- unreadable: OCR/필기 판독 불가.
- not_attempted: 시도 자체가 없음 (빈 줄, 포기 표시).

한 풀이에 wrong이 여러 개일 수 있습니다 (독립 오류 여러 건).
표기 실수는 관대하게 (± 방향, 괄호 위치 정도의 사소한 실수는 ok).

[감점형 루브릭 — 100점 만점 (25점 × 4기준)]
- 개념 사용 (25점): 문항이 요구한 핵심 개념을 정확히 사용했는가.
    예: 역함수 존재 조건 확인, 합성함수 순서, 정의역/공역/치역 구분.
- 근거 제시 (25점): 판단 이유를 말이나 수식으로 설명했는가.
    예: "일대일이므로 역함수 존재", "x=-2를 g에 대입" 같은 근거 서술.
- 계산·표현 (25점): 식 변형, 대응, 좌표 표현이 정확한가.
    예: 정확한 대입, 정확한 산술, 좌표계 표현.
- 결론 (25점): 최종 답을 명확히 썼는가.
    예: "답: 13"처럼 결론이 뚜렷하며, 정답과 일치.
{scale_section}
[감점 규칙 — 반드시 지킬 것]
1. 감점 사유마다 근거와 함께 나눠서 기록. 한 번에 25점 다 깎지 마세요.
2. 감점마다 `reason` (사유), `line_ref` (몇 번째 줄), `evidence` (원문 인용) 반드시 명시.
   - evidence는 학생 풀이에서 문자 그대로 발췌. 지어내면 안 됩니다.
   - 특정 줄 아닌 전반 감점이면 line_ref=0.
3. 감점 없으면 `deductions: []` (빈 배열), score = 25.
4. 앞 오류에서 전파된 결과는 중복 감점 금지 (steps depends와 연동).
5. 각 기준 score = max(0, 25 - 감점합계). 음수 나오지 않게 자를 것.
6. total_score = 4기준 score 합계.

{_GOOD_EXAMPLE}

{_BAD_EXAMPLE}

[observed_errors]
관찰한 구체적 실수를 짧은 문장으로 나열 (예: "합성함수 순서 뒤바꿈", "정의역 제한 미확인").
같은 실수 여러 번은 한 번만.

반드시 grade_student_solution tool로만 응답하세요."""


def build_retry(base_prompt: str, failed_list: str, ocr_base: str) -> str:
    """evidence 검증 실패 시 재시도 프롬프트"""
    return base_prompt + f"""

[⚠ 재시도 사유]
이전 응답의 다음 evidence 인용이 학생 풀이 원문에 존재하지 않습니다:
{failed_list}

evidence 필드는 반드시 아래 학생 풀이 원문에서 **문자 그대로** 발췌하세요.
지어내거나 요약/의역하지 마세요. 인용할 게 없으면 감점 자체를 취소하세요 (근거 없는 감점 금지).

[학생 풀이 원문 (문자 그대로 대조 대상)]
{ocr_base}
"""
