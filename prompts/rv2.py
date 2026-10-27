# -*- coding: utf-8 -*-
"""
prompts/rv2.py — R v2 자료(엑셀)에서 문항별 채점 프롬프트를 조립한다. (hongsu 이식본)

이용오 선생님 키트 prompt_builder.py(kit-1.2-20261022)를 그대로 옮긴 것.
차이는 두 가지뿐:
  (1) 자료 출처: materials.json 대신 rv2_codebook.get() (서버 시작 시 엑셀 직접 로드, 구조 동일)
  (2) 출력 방식: "JSON만 출력" 지시 대신 rv2_diagnose tool_use 강제
      (실험에서 Opus가 JSON 앞 수식 중괄호 때문에 파싱 실패한 사례가 있어, 구조화 출력으로 원천 차단)
  나머지 프롬프트 문구·규칙·순서는 키트와 동일.

--- 아래는 키트 원본 설명 ---
prompt_builder.py — R v2 자료(materials.json)에서 문항별 채점 프롬프트를 조립한다.

설계 원칙:
  - 프롬프트에 하드코딩된 코드북 내용은 없다. 전부 materials.json(= 오세준 교수님
    엑셀의 변환본)에서 읽는다. 엑셀이 바뀌면 convert 스크립트만 다시 돌리면 된다.
  - mode="grade_only"   : 1·2단계용. 코드북 없이 줄별 정오 판정 + 이유 서술만.
  - mode="with_codebook": 3단계용. 이 문항에 걸린 R코드만 발췌 주입 + J 수준 + 기권 규칙.
  - 같은 답안을 두 모드로 돌린 차이 = 전문가 코드북의 기여분 (절제 실험).

사용:
    from prompt_builder import load_materials, build_prompt
    mats = load_materials()
    p = build_prompt(mats, "INV-R-01", mode="with_codebook")
    # p["system"], p["user_template"], p["version"], p["allowed_codes"]
    # 실제 호출 시 p["user_template"]의 {STUDENT_ANSWER} 를 답안으로 치환
"""
import re


CODEBOOK_VERSION = "R-v2"
BUILDER_VERSION = "kit-1.2-20261022"  # v1.2: 33건 개발셋 교정 패치 (J 6규칙, C-1~C-5)
PORT_VERSION = "hongsu-rv2-1.0"       # hongsu 이식본 버전

# v1.2 패치 — 33건 개발셋(전문가 라벨)으로 교정된 추가 판정 규칙.
# 주의: 사후 적합된 규칙이므로 홀드아웃(신규 합성답안)에서 검증 전까지는 잠정.
V12_CODE_RULES = """<추가 판정 규칙 v1.2>
C-1 (R1/R5 경계): '한 입력에 두 값' 문제를 인지하고 다루다가 틀리면 R1,
    역함수 존재 조건 자체를 인지하지 못한 채 통과·부정하면 R5.
C-2 (± 처리): ±를 최종 답으로 제시하면 R6. 기준 풀이의 중간 단계와 일치하는 ±는 ok.
    중단된 답안은 쓰인 줄까지만 판정하고, 쓰인 줄이 모두 옳으면 코드 없이
    미완(response_complete=false)으로만 기록한다.
C-3 (근거 부재는 코드가 아님): 이 원칙이 기준 판정표의 개별 문구에 우선한다.
    반례·근거 없이 결론만 쓴 답안은 코드 없음(J와 요구충족으로만 기록).
    반례를 세우려다 틀린 경우에만 해당 코드를 부여한다.
C-4 (대안 정답): 문항 정답에 괄호로 명시된 대안 결론도 정답으로 인정한다.
C-5 (복수 후보): 한 줄에 코드가 둘 이상 성립하면 기준 판정표 해당 줄의 규칙을
    main으로 하고 나머지는 code_candidates에 병기한다.
</추가 판정 규칙 v1.2>"""

V12_J_RULES = """J 판정 세부 규칙 (v1.2):
1. 조건·이유를 명시했거나, 옳은 수학적 전개가 문항 목표의 조건을 완결되게 담으면 J2
   (유효한 반례 제시 포함 — 말로 풀지 않아도 된다).
2. 식 변형 전개는 있으나 조건 설명이 빠지면 J1.
3. 관련 개념을 말로 일부 언급하면 J1 (이유가 틀렸어도 개념을 다루면 J1).
4. 값 대입·읽기만 나열하거나, 조건 없는 상투어("대칭이니까", "반대로")뿐이면 J0.
   문항이 설명을 요구하는데 아무 서술이 없으면 J0.
5. 최종식 한 줄만 쓴 답안: 조건·부호 선택이 옳게 반영돼 있으면 J1, ± 미해소·미완이면 J0.
6. 중단 답안(결론 미도달)은 J0."""


def load_materials() -> dict:
    """키트와 같은 구조의 dict (엑셀에서 로드)."""
    import rv2_codebook
    return rv2_codebook.get()


def _problem(mats: dict, problem_id: str) -> dict:
    for p in mats["problems"]:
        if p["problem_id"] == problem_id:
            return p
    raise KeyError(f"problem_id not found: {problem_id}")


def _lines_for(mats: dict, problem_id: str) -> list:
    return [r for r in mats["line_judgment"] if r["problem_id"] == problem_id]


def allowed_codes(mats: dict, problem_id: str) -> list:
    """이 문항에서 판정 가능한 R코드 집합 = 01시트 주_R코드 + 06시트 wrong일_때에 등장하는 코드."""
    codes = set(re.findall(r"R\d+", _problem(mats, problem_id).get("주_R코드", "")))
    for r in _lines_for(mats, problem_id):
        codes |= set(re.findall(r"R\d+", r.get("wrong일_때", "") or ""))
    return sorted(codes, key=lambda c: int(c[1:]))


def _codebook_block(mats: dict, codes: list) -> str:
    rows = [r for r in mats["r_codebook"] if r["code"] in codes]
    out = []
    for r in rows:
        out.append(
            f"{r['code']}. {r['name']}\n"
            f"- 정의: {r['definition']}\n"
            f"- 포함: {r['include']}\n"
            f"- 제외: {r['exclude']}\n"
            f"- 예시: {r['example']}"
        )
    return "\n\n".join(out)


def _j_block(mats: dict) -> str:
    rows = mats["j_codebook"]
    out = [f"{r['code']} ({r['name']}): {r['criterion']} 예: \"{r['example']}\"" for r in rows]
    out.append("내용 오류와 독립으로 판정하며, 정답에도 붙인다. "
               "문항이 이유를 요구하지 않으면 not_applicable.")
    out.append(V12_J_RULES)
    return "\n".join(out)


def _criteria_block(mats: dict, problem_id: str, with_codes: bool) -> str:
    out = []
    for r in _lines_for(mats, problem_id):
        line = f"기준{r['line_no']}. {r['기준_풀이']}"
        w = (r.get("wrong일_때") or "").strip()
        if w and with_codes:
            line += f"   — wrong일 때: {w}"
        out.append(line)
    return "\n".join(out)


def _rules_block(mats: dict, with_codes: bool) -> str:
    skip_for_grade_only = {"5", "9", "10"}  # 코드 선택·오탐검사·피드백 관련 규칙은 3단계부터
    out = []
    for r in mats["prompt_rules"]:
        if not with_codes and str(r["번호"]) in skip_for_grade_only:
            continue
        out.append(f"{r['번호']}. {r['규칙']} — {r['설명']}")
    return "\n".join(out)


SCHEMA_GRADE_ONLY = """{
 "lines": [{"no": 1, "student_text": "...", "matched_criterion": 2 또는 null,
            "judgment": "ok|wrong|depends", "quote": "wrong이면 원문 인용, 아니면 null",
            "reason": "wrong이면 무엇이 왜 틀렸는지 한 문장", "depends_on": 원인 줄 번호 또는 null}],
 "math_result_correct": true,
 "input_status": "confirmed|not_attempted|unreadable",
 "notes": ""
}"""

SCHEMA_WITH_CODEBOOK = """{
 "lines": [{"no": 1, "student_text": "...", "matched_criterion": 2 또는 null,
            "judgment": "ok|wrong|depends", "code": "R코드 또는 null",
            "quote": "wrong이면 원문 인용, 아니면 null",
            "reason": "판정 근거 한 문장", "depends_on": 원인 줄 번호 또는 null}],
 "main_error_code": "첫 독립 wrong 줄의 R코드, 없으면 null",
 "justification_level": "J0|J1|J2|not_applicable",
 "math_result_correct": true,
 "input_status": "confirmed|not_attempted|unreadable",
 "review_needed": false,
 "review_reason": ""
}"""


TOOL_NAME = "rv2_diagnose"

_LINE_PROPS = {
    "no": {"type": "integer", "description": "줄 번호 (학생 번호 매김/줄바꿈 순서, 1부터)"},
    "student_text": {"type": "string"},
    "matched_criterion": {"type": ["integer", "null"], "description": "대응되는 기준 번호, 없으면 null"},
    "judgment": {"type": "string", "enum": ["ok", "wrong", "depends"]},
    "quote": {"type": ["string", "null"], "description": "wrong이면 학생 답안 원문 인용, 아니면 null"},
    "reason": {"type": ["string", "null"]},
    "depends_on": {"type": ["integer", "null"], "description": "depends면 원인 줄 번호"},
}


def tool_schema(mode: str) -> dict:
    """SCHEMA_* 텍스트와 같은 필드를 tool input_schema로."""
    line_props = dict(_LINE_PROPS)
    props = {
        "math_result_correct": {"type": "boolean"},
        "input_status": {"type": "string", "enum": ["confirmed", "not_attempted", "unreadable"]},
    }
    required = ["lines", "math_result_correct", "input_status"]
    if mode == "with_codebook":
        line_props["code"] = {"type": ["string", "null"], "description": "R코드 또는 null"}
        props.update({
            "main_error_code": {"type": ["string", "null"], "description": "첫 독립 wrong 줄의 R코드, 없으면 null"},
            "justification_level": {"type": "string", "enum": ["J0", "J1", "J2", "not_applicable"]},
            "review_needed": {"type": "boolean"},
            "review_reason": {"type": "string"},
        })
        required += ["main_error_code", "justification_level", "review_needed"]
    else:
        props["notes"] = {"type": "string"}
    props["lines"] = {"type": "array", "items": {"type": "object", "properties": line_props,
                                                 "required": ["no", "student_text", "judgment"]}}
    return {
        "name": TOOL_NAME,
        "description": "학생 답안을 줄별로 판정하고 (코드북 모드면) R코드·J수준을 기록한다.",
        "input_schema": {"type": "object", "properties": props, "required": required},
    }


def build_prompt(mats: dict, problem_id: str, mode: str = "with_codebook") -> dict:
    assert mode in ("grade_only", "with_codebook")
    p = _problem(mats, problem_id)
    with_codes = (mode == "with_codebook")
    codes = allowed_codes(mats, problem_id) if with_codes else []

    system = (
        "당신은 고등학교 공통수학2 「함수와 그래프」 역함수 단원의 서술형 채점자입니다. "
        "아래에 주어진 자료만 사용해 판정하며, 자료에 없는 오류 코드를 만들지 않습니다. "
        "모든 wrong 판정은 학생 답안 원문 인용으로 뒷받침해야 합니다."
    )

    parts = []
    if with_codes:
        parts.append("<이 문항의 오류 코드>\n" + _codebook_block(mats, codes) + "\n</이 문항의 오류 코드>")
        parts.append("<근거 수준>\n" + _j_block(mats) + "\n</근거 수준>")

    parts.append(
        "<문항>\n"
        f"{p['문항']}\n"
        f"정답: {p['정답']} / 요구 항목: {p['required_evidence']}\n"
        "</문항>"
    )
    parts.append("<줄별 기준 판정표>\n" + _criteria_block(mats, problem_id, with_codes) + "\n</줄별 기준 판정표>")
    parts.append("<판정 규칙>\n" + _rules_block(mats, with_codes) + "\n</판정 규칙>")
    if with_codes:
        parts.append(V12_CODE_RULES)

    steps = [
        "1) 학생 답안을 줄로 나눈다 (학생의 번호 매김 또는 줄바꿈을 따른다)",
        "2) 각 줄을 기준 판정표의 기준에 대응시키고 ok/wrong/depends 판정"
        " — 어떤 기준에도 대응되지 않는 줄은 수학적 정오만 판정",
    ]
    if with_codes:
        steps += [
            "3) wrong 줄에는 원문 인용과 함께 코드 부여."
            " 기준 판정표로 설명되지 않는 오류는 코드를 억지로 고르지 말고 review_needed=true",
            "4) 풀이 전체의 근거 수준(J) 판정",
        ]
        schema = SCHEMA_WITH_CODEBOOK
    else:
        steps += ["3) wrong 줄에는 원문 인용과 함께 무엇이 왜 틀렸는지 한 문장으로 서술"]
        schema = SCHEMA_GRADE_ONLY
    parts.append("판정 절차:\n" + "\n".join(steps))
    parts.append(f"반드시 {TOOL_NAME} 도구로만 응답하세요. 필드 구조:\n" + schema)
    parts.append("<학생 답안>\n{STUDENT_ANSWER}\n</학생 답안>")

    return {
        "system": system,
        "user_template": "\n\n".join(parts),
        "version": f"{CODEBOOK_VERSION}-{mode}-{BUILDER_VERSION}-{PORT_VERSION}-x{mats.get('xlsx_sha8', 'na')}",
        "allowed_codes": codes,
        "problem_id": problem_id,
    }


if __name__ == "__main__":
    import sys
    mats = load_materials()
    pid = sys.argv[1] if len(sys.argv) > 1 else "INV-R-01"
    mode = sys.argv[2] if len(sys.argv) > 2 else "with_codebook"
    built = build_prompt(mats, pid, mode)
    print(f"# version: {built['version']} / allowed: {built['allowed_codes']}\n")
    print("[SYSTEM]\n" + built["system"] + "\n")
    print("[USER]\n" + built["user_template"])
