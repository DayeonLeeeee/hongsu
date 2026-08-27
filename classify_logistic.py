"""
오답유형 분류 - 다항 로지스틱 회귀(Multinomial Logistic Regression) 버전
=====================================================================
기존 error_profiles.py 의 euclidean() 기반 classify_error()를 대체하는 모듈.

*** 변경하지 않는 것 ***
- FEATURE_KEYS (13차원 feature 정의) - 그대로 재사용
- ERROR_LABELS, MISSING_CONCEPTS, FEEDBACK_TEMPLATES, RECOMMENDED_FLOW
  (H1~H10의 정의·판정기준·피드백 - 오세진 교수님 설계, 절대 수정 안 함)
- UNIT_H_RELEVANCE (단원별 관련 H코드)
- 반환 스키마(primary_h, secondary_h, distribution 등) - S4 검증 레이어가
  이 스키마를 그대로 소비하므로 필드명을 유지해 하위 파이프라인 수정 불필요

*** 바뀌는 것 ***
- "수작업 profile과의 유클리드 거리" → "라벨 데이터로 학습한 확률모형"
- 단원 페널티(×5.0), is_typical(<1.0) 같은 매직넘버 → 확률 마스킹, 확률값 그대로 사용

*** 왜 이 모형인가 (요약) ***
1. MLE(최대우도추정) 기반이라 계수가 데이터로 추정됨 - 수작업 추정치 문제 해결
2. 음의 로그우도가 볼록함수(convex)라 항상 전역 최적해로 수렴 - 학습 안정성 보장
3. 라벨 수십 건 수준에서도 동작 - 9~10월 초기 교사 라벨만으로 시작 가능
4. 기존 13차원 feature 구조를 그대로 재사용 - LLM feature 추출 파이프라인 변경 불필요
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# 기존 모듈에서 그대로 재사용 (H1~H10 정의/라벨은 이 파일에서 절대 재정의하지 않음)
from error_profiles import FEATURE_KEYS, ERROR_LABELS, UNIT_H_RELEVANCE

MODEL_PATH = Path("h_classifier_logreg.json")


# ---------------------------------------------------------------------------
# 1. feature 변환
# ---------------------------------------------------------------------------

def vector_to_array(student_vector: dict) -> list[float]:
    """딕셔너리 → sklearn 입력 배열.
    feature 누락 시 0.5(중립값)로 채움 - 기존 euclidean()의 관례와 동일하게 유지."""
    return [float(student_vector.get(k, 0.5)) for k in FEATURE_KEYS]


# ---------------------------------------------------------------------------
# 2. 학습
# ---------------------------------------------------------------------------

def train_classifier(labeled_examples: list[dict]) -> LogisticRegression:
    """
    labeled_examples: [{"vector": {...13개 feature...}, "h_code": "H1"}, ...]
    교사 1차 라벨(9~10월 수집분)이 쌓이면 이 함수로 학습/재학습.

    [주의: 소표본 완전분리(perfect separation) 문제]
    특정 feature 하나만으로 클래스가 완벽히 갈리면 계수가 발산할 수 있음.
    클래스당 라벨이 5개 미만이면 정규화(C값)를 자동으로 강하게 건다.
    """
    X = [vector_to_array(ex["vector"]) for ex in labeled_examples]
    y = [ex["h_code"] for ex in labeled_examples]

    counts = {h: y.count(h) for h in set(y)}
    missing_classes = set(ERROR_LABELS) - set(counts)
    if missing_classes:
        print(f"[경고] 아직 라벨이 하나도 없는 H코드: {sorted(missing_classes)} "
              f"- 해당 유형은 이번 학습에서 제외되며 예측 후보에서도 빠집니다.")

    min_count = min(counts.values()) if counts else 0
    if min_count < 5:
        print(f"[경고] 클래스당 라벨 수가 적습니다({counts}). "
              f"과적합/발산 방지를 위해 정규화를 강하게 겁니다(C=0.3).")
        clf = LogisticRegression(multi_class="multinomial", C=0.3, max_iter=3000)
    else:
        clf = LogisticRegression(multi_class="multinomial", C=1.0, max_iter=3000)

    clf.fit(X, y)

    # 재현성을 위해 계수 저장 (프롬프트 버전관리와 같은 이유 - 어떤 모델로 어떤 결과가 나왔는지 추적)
    MODEL_PATH.write_text(json.dumps({
        "classes": clf.classes_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "n_train": len(y),
        "class_counts": counts,
    }, ensure_ascii=False, indent=2))

    return clf


# ---------------------------------------------------------------------------
# 3. 분류
# ---------------------------------------------------------------------------

def classify_error_logreg(student_vector: dict, unit: str = "",
                           clf: LogisticRegression | None = None) -> dict:
    """
    기존 classify_error()와 동일한 필드명을 반환 -> 호출부(S4 검증 레이어 등) 수정 불필요.
    clf를 넘기지 않으면 저장된 계수(h_classifier_logreg.json)를 로드해서 사용.
    """
    if clf is None:
        clf = _load_saved_classifier()

    x = np.array([vector_to_array(student_vector)])
    proba = clf.predict_proba(x)[0]
    classes = clf.classes_

    # 기존의 "단원 무관 H코드에 거리 5배 페널티" -> "확률에 마스킹 후 재정규화"로 대체.
    # 로직은 동일(단원과 무관한 유형의 순위를 낮춤)하되, 결과가 여전히 확률 분포로 유지됨.
    relevant = UNIT_H_RELEVANCE.get(unit, [])
    if relevant:
        mask = np.array([1.0 if h in relevant else 0.05 for h in classes])
        proba = proba * mask
        proba = proba / proba.sum()

    ranked = sorted(zip(classes, proba), key=lambda t: -t[1])
    primary_h, primary_p = ranked[0]
    secondary_h, secondary_p = ranked[1]

    return {
        "primary_h": primary_h,
        "primary_label": ERROR_LABELS[primary_h],
        "primary_prob": round(float(primary_p), 4),
        "secondary_h": secondary_h,
        "secondary_label": ERROR_LABELS[secondary_h],
        "secondary_prob": round(float(secondary_p), 4),
        "prob_gap": round(float(primary_p - secondary_p), 4),
        # 기존 is_typical(거리<1.0, 근거 없는 임계값)을 확률 기준으로 대체 - 훨씬 직관적
        "is_confident": bool(primary_p > 0.5),
        "distribution": {h: round(float(p), 4) for h, p in zip(classes, proba)},
    }


def _load_saved_classifier() -> LogisticRegression:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH}가 없습니다. train_classifier()로 먼저 학습해야 합니다. "
            f"(교사 라벨이 아직 없다면 이 분류기는 아직 사용할 수 없는 게 정상입니다.)"
        )
    saved = json.loads(MODEL_PATH.read_text())
    clf = LogisticRegression(multi_class="multinomial")
    clf.classes_ = np.array(saved["classes"])
    clf.coef_ = np.array(saved["coef"])
    clf.intercept_ = np.array(saved["intercept"])
    return clf


if __name__ == "__main__":
    # 최소 동작 예시 (실데이터 없이 구조만 확인용 - 실제 학습에는 쓰지 말 것)
    demo_examples = [
        {"vector": {"checked_uniqueness": 0.1, "checked_definition_domain": 0.5,
                     "checked_one_to_one": 0.3, "has_reasoning": 0.4}, "h_code": "H1"},
        {"vector": {"checked_uniqueness": 0.6, "checked_definition_domain": 0.1,
                     "checked_one_to_one": 0.4, "has_reasoning": 0.4}, "h_code": "H2"},
    ]
    print("[데모] 이 예시는 구조 확인용이며, 실제 분류에는 9~10월 교사 라벨 데이터가 필요합니다.")
