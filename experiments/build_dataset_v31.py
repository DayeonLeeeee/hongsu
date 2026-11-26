# coding: utf-8
"""
자료 엑셀을 코드북 R v3.1(2026-11-26, 오세준 교수님)에 맞춘 사본으로 만든다.

  python experiments/build_dataset_v31.py            → materials/dataset_rv31.xlsx

- 원본 materials/dataset_rv2.xlsx(키트와 같은 파일, R-v3)는 건드리지 않는다.
- 바꾼 칸은 모두 새 시트 'CHANGES_v3.1'에 (시트, 키, 열, 이전, 이후, 출처)로 남긴다.
- 바꾸는 것은 코드북 v3.1 문서에 적힌 것만이다:
    02_R_CODEBOOK   R1~R11(R4 제외) 이름·정의·포함·제외·예시·근거를 v3.1 표 문구로, 주의(caution) 열 추가
                    R4는 1차 적용 보류(v3.1 2.) — 이름에 '(1차 보류)'
    09_PROMPT_RULES 판정 규칙 번호를 v3.1 7.에 맞춤: 11에 단서, 12 = 주 코드 선정(신설).
                    Rv3의 12(미완 처리)는 v3.1에서 '답안 상태' 절로 옮겨졌으므로 번호를 '답안 상태'로 (문구는 그대로)
    06_LINE_JUDGMENT 붙임 1: 문항 2 기준 ④, 문항 7 기준 ③·④
    01_PROBLEMS     문항 7 주_R코드에서 R4 제외 (R4 보류)
    04_SYNTHETIC_ANSWERS 붙임 1: 문항 3 대표 오답 B(A-03-B) 비고에 단서 (프롬프트에는 들어가지 않음)
    00_README       코드북 버전 R-v3.1
- 아직 반영하지 않은 것: 문항 1 수치 교체(2차 적용부터), 미완 처리 문구(교수님 답 대기),
  프롬프트 코드의 C-1·판정 절차 4번 문장(교수님 답 대기).
"""
import os
import shutil

import openpyxl

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "materials", "dataset_rv2.xlsx")
DST = os.path.join(BASE, "materials", "dataset_rv31.xlsx")
SRC_DOC = "코드북 R v3.1 (2026-11-26, 오세준)"

# 코드북 v3.1 4. 내용 오류 코드 표 문구 그대로 (R11 주의는 표 아래 'R7과의 경계' 문단)
V31_CODES = {
    "R1": {
        "name": "방향을 바꾼 대응의 함수성 미확인",
        "정의": "화살표의 방향을 반대로 바꾼 대응이 집합 Y에서 집합 X로의 함수가 되는지 확인하지 않거나, 확인하고도 함수라고 잘못 판단한다",
        "포함": "대응하는 원소가 2개인 것이 있는데 함수라고 함 / 대응하는 원소가 없는 것이 있는데 함수라고 함 / 방향을 바꾼 대응을 살피지 않고 ‘반대로 하면 된다’로 결론",
        "제외": "방향을 바꾼 대응이 함수가 아님을 옳게 보인 경우. 둘 중 하나만 들어도 충분한 근거임",
        "예시": "“화살표를 반대로 하면 함수가 된다”",
        "근거": "Bayazit & Gray(2004) 107~109쪽 — 방향을 바꾼 대응이 함수인지 놓친 답을 보고"
    },
    "R2": {
        "name": "일대일 조건 오류 및 일대일함수·일대일대응 혼동",
        "정의": "일대일함수·일대일대응의 조건을 잘못 세우거나(필요한 조건을 더 약한 조건으로 대체하는 것 포함) 두 개념을 혼동한다",
        "포함": "구간별 조건만 확인하고 전체 정의역에서의 일대일을 확인하지 않음 / 기울기가 0이 아니면 일대일이라고 판단 / “함숫값이 겹치지 않으므로 일대일대응이다” / 공역에 남는 원소를 확인하지 않고 일대일대응이라고 함",
        "제외": "“일대일함수지만 일대일대응은 아니다. g⁻¹:g(X)→X는 가능하다”는 정답임",
        "예시": "“두 조각의 기울기가 0이 아니므로 일대일대응이다”",
        "근거": "Bayazit & Gray(2004) · 신규 답안 H-09 · H-13 · H-23",
        "주의": "포함 항목과 문구가 겹쳐도 결론이 참이면 코드가 아님(규칙 11). ‘함숫값이 겹치지 않으므로’는 근거 수준 J1의 예시와도 문구가 가까움. 결론의 정오를 먼저 보고 판단"
    },
    "R3": {
        "name": "공역과 치역 구분 누락",
        "정의": "실제로 나온 함숫값의 집합과 목표 집합을 같은 것으로 본다",
        "포함": "치역을 공역 전체로 씀 / 남는 원소가 있는지 확인하지 않음",
        "제외": "문항이 치역을 묻지 않았는데 쓰지 않은 경우 / 치역=공역 관계를 옳게 알고 있으나 이를 미지수 결정에 활용하지 못한 경우(연결 실패)는 R3이 아님. 코드 없이 기록함",
        "예시": "g(X)={a,b,c}인데 “치역은 {a,b,c,d}이다”",
        "근거": "CVI에서 우선 검증 · 연결 실패 제외는 H-04 · H-16 · H-20"
    },
    "R5": {
        "name": "역함수 존재 조건의 잘못된 일반화",
        "정의": "역함수의 존재 조건을 잘못 일반화한 진술을 명시한다",
        "포함": "“함수니까 역함수가 있다” / “모든 함수에는 역함수가 있다” / “식으로 쓸 수 있으면 역함수가 있다”",
        "제외": "올바른 절차를 보였으나 존재 조건을 따로 쓰지 않은 것 / 그런 진술을 쓰지 않은 답안에 추론으로 붙이지 않음. 오류 찾기 문항에서 ‘없다’고만 한 답은 그 자체로 R5가 아님(규칙 12)",
        "예시": "“함수니까 역함수가 있다”",
        "근거": "조정수(2024)가 존재·비존재를 대비한 문항을 후속 과제로 제안"
    },
    "R6": {
        "name": "역함수 식 변형 오류",
        "정의": "역함수의 식을 세우고 bx+c 꼴로 쓰기까지의 변형 과정에서 이항·부호·근호 앞 부호 선택을 틀린다",
        "포함": "x=2y−3 → y=(x−3)/2 이항 부호 / (x−3)/a를 bx+c 꼴로 펼치며 상수항의 부호를 반대로 읽음 / y=±√(x−1)로 두 가지를 함께 씀 / 근호 앞 부호를 원래 정의역과 반대로 고름 / 변형 과정 안의 분수 계산 실수",
        "제외": "x와 y를 바꾸지 않고 원래 변수로 옳게 푼 경우는 정답임 / 변형이 끝난 뒤의 대입·산술 실수. 앞 줄이 맞으면 코드 없이 wrong, 앞 줄이 틀렸으면 depends",
        "예시": "f⁻¹(x)=±√(x−1)",
        "근거": "조정수(2024) 527~536쪽 — ±√(1−x)를 다룸"
    },
    "R7": {
        "name": "정의역과 치역의 교환 누락",
        "정의": "역함수의 정의역이 원래 함수의 치역이고 치역이 원래 함수의 정의역이라는 교환을 하지 않거나 뒤바꾼다",
        "포함": "f⁻¹의 정의역을 원래 함수의 정의역으로 씀 / 제한된 정의역을 무시하고 역함수를 구함",
        "제외": "범위를 아예 쓰지 않은 답은 이 코드가 아님. 문항이 범위를 요구했는지 먼저 확인",
        "예시": "f:[0,∞)→[1,∞)인데 “f⁻¹(x)=√(x−1), 정의역 [0,∞)”",
        "근거": "조정수(2024) 527~536쪽"
    },
    "R8": {
        "name": "y=x 대칭과 좌표 교환 오류",
        "정의": "y=x 대칭을 점의 좌표 교환 (a,b)→(b,a)로 연결하지 못하거나, 대칭축을 x축·y축·원점으로 잘못 쓴다",
        "포함": "y축 대칭으로 (−2,5)를 씀 / x축 대칭으로 (2,−5)를 씀 / 축을 밝히지 않은 채 좌표를 틀리게 씀",
        "제외": "“y=x 대칭이므로 (5,2)”처럼 짧아도 결과가 맞으면 정답임",
        "예시": "(2,5)의 대칭점을 (−2,5)로 씀",
        "근거": "Breen 외(2017) 93~99쪽 — 대칭을 말한 39건 중 y=x를 정확히 말한 것은 9건"
    },
    "R9": {
        "name": "교점 성질의 과잉 일반화",
        "정의": "두 그래프가 y=x에 대칭이라는 사실을, 교점이 항상 y=x 위에 있다는 주장으로 일반화한다",
        "포함": "“대칭이니까 교점은 항상 y=x 위이다” / 반례를 확인하지 않고 일반 명제로 결론 / 교점 성질을 잘못 판단하여 부적절한 반례를 제시",
        "제외": "대칭이동 그리기만 틀린 경우는 R8, 교점 계산 실수는 R6 / 반례를 제시하지 못했다는 사실만으로 R9를 부여하지 않음",
        "예시": "f(x)=1−x=f⁻¹(x)에서 (0,1)은 두 그래프 위이나 y=x 위가 아님",
        "근거": "천재(홍진곤) 122쪽이 이 주장을 「추론」 활동으로 제시 · YBM(류희찬) Ⅲ-3 본문 탐구 질문",
        "주의": "x가 커질수록 함숫값도 커지는 함수에서는 이 주장이 참임. 조건 없이 일반화한 경우에만 오류이며, 전칭 표현(‘항상’·‘무조건’)의 유무가 갈림길임"
    },
    "R10": {
        "name": "역함수 표기를 역수로 읽음",
        "정의": "f⁻¹을 f의 역수 1/f로 읽는 표기 해석 오류",
        "포함": "f⁻¹(8) = 1/f(8) / f⁻¹(x) = 1/(3x+2) / f⁻¹(8) = 1/8",
        "제외": "식 변형 중의 분수 계산 실수는 R6 / 학생 본인이 역수로 읽은 경우에만 씀. 오류 찾기 문항에서 제시된 풀이의 오류를 다른 종류로 오진한 것은 R10이 아님",
        "예시": "f(x)=3x+2에서 “f⁻¹(8)=1/26”",
        "근거": "서고 4편에서 직접 보고를 확인하지 못함. 문헌 보강 필요"
    },
    "R11": {
        "name": "정의역 제한이 선택임을 미인식",
        "정의": "역함수가 존재하도록 정의역을 제한할 때, 제한 방법이 여러 가지이고 그에 따라 역함수가 달라진다는 것을 인식하지 못한다",
        "포함": "x²+1에서 x≥0만이 유일한 제한이라고 씀 / 제한을 바꾸면 역함수가 달라진다는 것을 부정 / “정답은 √(x−1) 하나”라고 단정",
        "제외": "한 가지 제한만 요구한 문항에서 하나만 쓴 경우는 정답",
        "예시": "“정의역을 x≥0으로 해야 하므로 답은 √(x−1)뿐이다”",
        "근거": "조정수(2024)가 정의역 제한을 대비한 문항을 후속 과제로 제안",
        "주의": "R7과의 경계: R7은 교환을 하지 않은 것이고, R11은 교환은 했으나 제한이 자의적임을 모르는 것임. x≤0으로 제한하면 역함수는 −√(x−1)이 됨"
    }
}

CODEBOOK_FIELDS = (("name", "name"), ("정의", "definition"), ("포함", "include"), ("제외", "exclude"),
                   ("예시", "example"), ("근거", "evidence"), ("주의", "caution"))

RULE_11_TAIL = "다만 어느 코드를 붙일지는 규칙 12에 따른다"
RULE_12 = ("코드 후보가 둘 이상일 때는 학생이 실제로 쓴 문장으로 인용할 수 있는 코드를 주 코드로 한다",
           "쓰지 않은 것(부재)으로만 추론되는 코드는 주 코드로 쓰지 않고, 필요하면 missing_criteria로 기록한다. "
           "오류 찾기 문항에서 결론이 틀린 답안은, 이유를 함께 썼으면 그 이유에 드러난 오개념의 코드를 쓴다. "
           "이유를 쓰지 않았으면 기준 답에 해당하는 줄의 코드를 붙이고 J0으로 기록한다")

LINE_EDITS = {  # (problem_id, line_no): 새 wrong일_때   — 붙임 1
    ("INV-R-02", 4): "c = 3/a로 읽으면 R6",
    ("INV-R-07", 3): "0이 아니기만 하면 된다고 보아 같은 부호 조건을 빠뜨리면 R2",
    ("INV-R-07", 4): "부등식의 방향·부호를 틀리면 R6 · ③이 틀렸으면 depends",
}


def _header(ws):
    return [c.value for c in ws[1]]


def build(src=SRC, dst=DST):
    shutil.copyfile(src, dst)
    wb = openpyxl.load_workbook(dst)
    log = []

    def setv(ws, row, col, new, key, colname):
        old = ws.cell(row, col).value
        if (old or "") != (new or ""):
            ws.cell(row, col).value = new
            log.append((ws.title, key, colname, old, new, SRC_DOC))

    # 02_R_CODEBOOK
    ws = wb["02_R_CODEBOOK"]
    head = _header(ws)
    if "caution" not in head:
        ws.cell(1, len(head) + 1).value = "caution"
        head.append("caution")
    col = {h: i + 1 for i, h in enumerate(head)}
    for r in range(2, ws.max_row + 1):
        code = ws.cell(r, col["code"]).value
        if code in V31_CODES:
            for k, h in CODEBOOK_FIELDS:
                if k in V31_CODES[code]:
                    setv(ws, r, col[h], V31_CODES[code][k], code, h)
        elif code == "R4":
            setv(ws, r, col["name"], "그래프에서 함수·역함수 판정 오류 (1차 보류)", code, "name")

    # 09_PROMPT_RULES
    ws = wb["09_PROMPT_RULES"]
    rows = {str(ws.cell(r, 1).value): r for r in range(2, ws.max_row + 1) if ws.cell(r, 1).value is not None}
    r11 = rows["11"]
    d = ws.cell(r11, 3).value or ""
    if RULE_11_TAIL not in d:
        setv(ws, r11, 3, d.rstrip(". ") + ". " + RULE_11_TAIL, "11", "설명")
    r12 = rows["12"]
    if ws.cell(r12, 2).value != RULE_12[0]:
        old = (ws.cell(r12, 2).value, ws.cell(r12, 3).value)
        setv(ws, r12, 1, "답안 상태", "12(Rv3 미완 처리)", "번호")
        nr = ws.max_row + 1
        ws.cell(nr, 1).value, ws.cell(nr, 2).value, ws.cell(nr, 3).value = 12, RULE_12[0], RULE_12[1]
        log.append((ws.title, "12", "규칙·설명", None, " — ".join(RULE_12), SRC_DOC + " 7. 규칙 12 신설"))
        # 번호 순서: 1~11, 12(신설), 답안 상태
        vals = [[ws.cell(r, c).value for c in range(1, 4)] for r in range(2, ws.max_row + 1)]
        vals.sort(key=lambda v: (isinstance(v[0], str), v[0] if not isinstance(v[0], str) else 99))
        for i, v in enumerate(vals):
            for c in range(3):
                ws.cell(i + 2, c + 1).value = v[c]

    # 06_LINE_JUDGMENT
    ws = wb["06_LINE_JUDGMENT"]
    h = {x: i + 1 for i, x in enumerate(_header(ws))}
    for r in range(2, ws.max_row + 1):
        k = (ws.cell(r, h["problem_id"]).value, ws.cell(r, h["line_no"]).value)
        if k in LINE_EDITS:
            setv(ws, r, h["wrong일_때"], LINE_EDITS[k], f"{k[0]} 기준 {k[1]}", "wrong일_때")

    # 01_PROBLEMS — R4 보류
    ws = wb["01_PROBLEMS"]
    h = {x: i + 1 for i, x in enumerate(_header(ws))}
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, h["problem_id"]).value == "INV-R-07":
            setv(ws, r, h["주_R코드"], "R2", "INV-R-07", "주_R코드")

    # 04_SYNTHETIC_ANSWERS — 문항 3 대표 오답 B 단서 (프롬프트에는 안 들어감)
    ws = wb["04_SYNTHETIC_ANSWERS"]
    h = {x: i + 1 for i, x in enumerate(_header(ws))}
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, h["answer_id"]).value == "A-03-B":
            setv(ws, r, h["비고"], "v3.1 단서: 이유를 함께 쓴 답안은 그 이유에 드러난 코드를 따른다(규칙 12). "
                                   "이 답안처럼 '없다'만 쓰면 R5·J0", "A-03-B", "비고")

    # 00_README
    ws = wb["00_README"]
    for r in range(1, ws.max_row + 1):
        k = ws.cell(r, 1).value
        if k == "코드북":
            setv(ws, r, 2, "R-v3.1 · 내용 오류 R1–R3, R5–R11 (R4는 1차 보류) · 근거 수준 J0–J2 · 답안 상태 4종", "코드북", "내용")
        if k == "개정":
            v = ws.cell(r, 2).value or ""
            if "R-v3.1" not in v:
                setv(ws, r, 2, v + " / R-v3.1 (2026-11-26, 오세준 교수님 코드북 R v3.1) — 판정 규칙 12(주 코드 선정) 신설·"
                     "규칙 11 단서, R2·R5·R6·R9·R10·R11 문구, 문항 2·7 기준표, R4 1차 보류. 미완 처리 문구는 그대로", "개정", "내용")

    # 바꾼 칸 기록
    if "CHANGES_v3.1" in wb.sheetnames:
        del wb["CHANGES_v3.1"]
    ch = wb.create_sheet("CHANGES_v3.1")
    ch.append(["시트", "키", "열", "이전", "이후", "출처"])
    for row in log:
        ch.append(list(row))
    wb.save(dst)
    return log


if __name__ == "__main__":
    log = build()
    print(f"{DST}: 바꾼 칸 {len(log)}개 (시트 CHANGES_v3.1)")
