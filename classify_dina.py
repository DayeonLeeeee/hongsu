"""
오답유형 분류 - DINA(Deterministic Input, Noisy AND gate) 방식
=====================================================================
인지진단모형(CDM) 계열 중 가장 널리 쓰이는 DINA를 이 프로젝트 규모에 맞게
단순화한 버전.

*** 변경하지 않는 것 ***
- ERROR_LABELS (H1~H10 이름) - 오세진 교수님 설계, 절대 수정 안 함
- Q_MATRIX의 내용은 '판정기준' 원문(.docx의 "필수 풀이 근거" 항목)을
  그대로 이분 지표(0/1)로 옮긴 것 - 판정기준 재해석이 아니라 표기법 변환

*** 표준 DINA와의 차이 (반드시 인지할 것) ***
표준 DINA는 "여러 문항 x 여러 학생" 데이터에서 학생별 잠재 속성 프로파일을
EM으로 추정하는 모형이다. 이 프로젝트는 "문항 1개 답안 1개를 바로 H코드로
진단"하는 용도이므로, 다음과 같이 단순화했다:
  - slip/guess를 EM이 아니라 라벨 비율로 직접 추정(method-of-moments)
  - 여러 문항에 걸친 결합 추정이 아니라 H코드별 독립 추정
이 단순화는 8월 v1과 소규모 파일럿(1개 학급) 규모에는 적합하지만,
데이터가 충분히 쌓이면(다학급·다문항) R의 CDM 패키지 등 표준 도구로
정식 EM 재추정을 검토해야 한다. -> 논문에 "간이 DINA(simplified DINA)"로
명시할 것을 권장.

*** 왜 이 모형인가 (요약) ***
1. 오세진 교수님의 판정기준이 이미 "A확인 AND B확인"처럼 결합조건 문장으로
   쓰여 있음 - DINA의 conjunctive(AND) 구조와 형식이 정확히 일치
2. 로지스틱 회귀(보상적 모형)와 달리, "필요조건 중 하나라도 빠지면
   그 오류가 성립"이라는 비보상적 논리를 수식 그대로 표현 가능
3. slip/guess가 "학생이 알면서도 실수로 그 오류처럼 보이는 경우"를
   명시적으로 모형화 - 손글씨 채점의 현실(급하게 풀다 실수)과 부합
"""

from __future__ import annotations
from error_profiles import ERROR_LABELS

# ---------------------------------------------------------------------------
# Q-matrix: 각 H코드가 성립하려면 '결여'되어 있어야 하는 체크 항목.
# 출처: .docx의 문항별 "필수 풀이 근거" 문구를 그대로 이분 지표로 인코딩.
# ---------------------------------------------------------------------------
Q_MATRIX: dict[str, list[str]] = {
    "H1": ["checked_uniqueness"],
    "H2": ["checked_definition_domain"],
    "H3": ["checked_definition_domain"],
    "H4": ["checked_one_to_one"],
    "H5": ["checked_composition_order"],
    "H6": ["checked_one_to_one"],
    "H7": ["checked_composition_order"],
    "H8": ["checked_definition_domain"],
    "H9": ["checked_domain_restriction"],
    "H10": ["has_reasoning", "used_criterion"],
}
# [확인 필요: 오세진 교수님 확인 필요]
# H7(역함수 식 변형 오류), H8(역함수 그래프 오류)은 지금 13차원 feature 중
# 정확히 대응하는 체크 항목이 없어 근사치로 매핑했습니다.
# "역함수 식 변형 절차를 순서대로 밟았는가", "점 (a,b)->(b,a) 대응을
# 명시했는가" 같은 전용 이분 지표를 추가할지 확인이 필요합니다.

DEFAULT_SLIP = 0.2   # 조건을 갖췄는데도(η=1) 실수로 그 오류처럼 보일 확률 (초기값)
DEFAULT_GUESS = 0.2  # 조건이 없는데도(η=0) 우연히 그 오류가 아닌 것처럼 보일 확률 (초기값)


# ---------------------------------------------------------------------------
# 1. 이상반응(ideal response) 계산 - 결합조건(AND) 논리
# ---------------------------------------------------------------------------

def compute_eta(observed_checks: dict[str, bool], h_code: str) -> int:
    """필요한 체크가 전부 '결여'되어 있으면 1(해당 오류 성립), 아니면 0.
    하나라도 체크가 '되어 있으면'(True) 그 오류는 성립하지 않음 - conjunctive AND."""
    required = Q_MATRIX.get(h_code, [])
    if not required:
        return 0
    return int(all(not observed_checks.get(c, False) for c in required))


# ---------------------------------------------------------------------------
# 2. 우도 계산 (slip/guess 보정)
# ---------------------------------------------------------------------------

def likelihood(observed_checks: dict[str, bool], h_code: str,
               slip: float = DEFAULT_SLIP, guess: float = DEFAULT_GUESS) -> float:
    """이 h_code가 실제 오류일 때, 지금 관측된 체크 패턴이 나올 확률."""
    eta = compute_eta(observed_checks, h_code)
    return (1 - slip) if eta == 1 else guess


# ---------------------------------------------------------------------------
# 3. 분류 (사후확률 - 균등 사전확률 가정)
# ---------------------------------------------------------------------------

def classify_error_dina(observed_checks: dict[str, bool], unit: str = "",
                         slip_guess: dict[str, dict[str, float]] | None = None) -> dict:
    """
    observed_checks: {"checked_uniqueness": True/False, ...}
        - 주의: 기존 euclidean 방식(0~1 연속값)과 달리 boolean 필요.
          S3 프롬프트가 "이 조건을 확인했는가: 예/아니오"로 판정하도록
          출력 스키마를 바꿔야 함 (다연 확인 필요 - 연속값 스키마와 별도 유지 or 통합 논의).
    slip_guess: {h_code: {"slip": .., "guess": ..}} - estimate_slip_guess()로
        추정한 값. 없으면 DEFAULT_SLIP/GUESS(초기값) 사용.
    """
    scores: dict[str, float] = {}
    for h in ERROR_LABELS:
        sg = (slip_guess or {}).get(h, {})
        s = sg.get("slip", DEFAULT_SLIP)
        g = sg.get("guess", DEFAULT_GUESS)
        scores[h] = likelihood(observed_checks, h, s, g)

    total = sum(scores.values()) or 1.0
    posterior = {h: round(v / total, 4) for h, v in scores.items()}
    ranked = sorted(posterior.items(), key=lambda t: -t[1])

    return {
        "primary_h": ranked[0][0],
        "primary_label": ERROR_LABELS[ranked[0][0]],
        "primary_prob": ranked[0][1],
        "secondary_h": ranked[1][0],
        "secondary_label": ERROR_LABELS[ranked[1][0]],
        "secondary_prob": ranked[1][1],
        "posterior": posterior,
    }


# ---------------------------------------------------------------------------
# 4. slip/guess 추정 (method-of-moments - 정식 EM의 단순화 버전, 위 docstring 참고)
# ---------------------------------------------------------------------------

def estimate_slip_guess(labeled_examples: list[dict]) -> dict[str, dict[str, float]]:
    """
    labeled_examples: [{"checks": {...}, "h_code": "H1"}, ...] (교사 1차 라벨, 9~10월)

    slip = P(η=1인데 실제 라벨 ≠ h)   # 조건을 갖췄는데 다른 걸로 라벨링됨
    guess = P(η=0인데 실제 라벨 = h)  # 조건이 없는데 그 라벨로 판정됨

    [주의] 라벨 수가 극히 적으면(H코드당 5개 미만) 이 추정치의 분산이 커서
    신뢰하기 어려움 - 그 경우 DEFAULT_SLIP/GUESS를 그대로 쓰는 편이 안전.
    """
    result = {}
    for h in ERROR_LABELS:
        eta1_total = eta1_wrong = eta0_total = eta0_right = 0
        for ex in labeled_examples:
            eta = compute_eta(ex["checks"], h)
            if eta == 1:
                eta1_total += 1
                if ex["h_code"] != h:
                    eta1_wrong += 1
            else:
                eta0_total += 1
                if ex["h_code"] == h:
                    eta0_right += 1

        slip = round(eta1_wrong / eta1_total, 3) if eta1_total >= 5 else DEFAULT_SLIP
        guess = round(eta0_right / eta0_total, 3) if eta0_total >= 5 else DEFAULT_GUESS
        result[h] = {
            "slip": slip, "guess": guess,
            "n_eta1": eta1_total, "n_eta0": eta0_total,  # 추정 신뢰도 확인용
        }
    return result


if __name__ == "__main__":
    # 최소 동작 예시 (구조 확인용)
    example_checks = {
        "checked_uniqueness": False,   # 확인 안 함 -> H1 가능성
        "checked_definition_domain": True,
        "checked_one_to_one": True,
        "has_reasoning": True,
        "used_criterion": True,
    }
    result = classify_error_dina(example_checks, unit="함수")
    print("[데모]", result)
    print("(실제 slip/guess는 9~10월 교사 라벨로 estimate_slip_guess()를 돌려 갱신할 것)")
