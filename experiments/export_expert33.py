# coding: utf-8
"""
엑셀 04_SYNTHETIC_ANSWERS(전문가 33건)를 run_batch.py가 읽는 CSV로 내보낸다.

  python experiments/export_expert33.py            → answers/expert_33.csv

라벨 통일(1-2)용으로 `첫_독립_오류_기준` 열을 하나 더 만든다.
처음에는 `첫_독립_오류`를 그대로 복사해 두므로, 학생 줄 번호와 기준표 번호가 다른 답안만
사람이 고치면 된다 (11/11 자료 기준: A-01-A, A-01-B, A-04-A, A-04-B. A-05-B, A-07-B는 확인 필요).
이미 파일이 있으면 덮어쓰지 않는다 (사람이 고친 라벨 보호). 새로 만들려면 --force.
"""
import argparse
import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

COLUMNS = ["answer_id", "problem_id", "유형", "학생_답안", "줄별_판정", "첫_독립_오류", "첫_독립_오류_기준",
           "error_code", "justification_level", "비고"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=os.path.join(BASE_DIR, "materials", "dataset_rv2.xlsx"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "answers", "expert_33.csv"))
    ap.add_argument("--force", action="store_true", help="있는 파일을 덮어쓴다")
    ap.add_argument("--converted-ids", default="A-05-D,A-06-A,A-06-C,A-08-A,A-09-B",
                    help="1-4 변환용 틀을 만들 답안 (answers/converted_5.csv)")
    args = ap.parse_args(argv)
    write_converted_template(args)

    if os.path.exists(args.out) and not args.force:
        print(f"이미 있음: {args.out} (사람이 고친 라벨을 지키려고 덮어쓰지 않음. 새로 만들려면 --force)")
        return args.out

    import openpyxl
    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    rows = list(wb["04_SYNTHETIC_ANSWERS"].iter_rows(values_only=True))
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    out_rows = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        d = {h: ("" if v is None else str(v)) for h, v in zip(head, r)}
        d["첫_독립_오류_기준"] = d.get("첫_독립_오류", "")
        out_rows.append({c: d.get(c, "") for c in COLUMNS})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)}건 → {args.out}")
    return args.out


def write_converted_template(args) -> None:
    """1-4용 틀: 원래 답안을 '원래_답안'에 두고 '학생_답안'은 비워 둔다. 사람이 풀이 형태로 채운다."""
    out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "converted_5.csv")
    if os.path.exists(out) and not args.force:
        print(f"이미 있음: {out}")
        return
    import openpyxl
    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    rows = list(wb["04_SYNTHETIC_ANSWERS"].iter_rows(values_only=True))
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    want = [x.strip() for x in args.converted_ids.split(",") if x.strip()]
    found = {}
    for r in rows[1:]:
        d = {h: ("" if v is None else str(v)) for h, v in zip(head, r)}
        if d.get("answer_id") in want:
            found[d["answer_id"]] = d
    cols = ["answer_id", "problem_id", "원래_답안", "학생_답안", "첫_독립_오류_기준", "error_code",
            "justification_level", "비고"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for aid in want:
            d = found.get(aid, {"answer_id": aid})
            w.writerow({"answer_id": aid, "problem_id": d.get("problem_id", ""), "원래_답안": d.get("학생_답안", ""),
                        "학생_답안": "", "첫_독립_오류_기준": d.get("첫_독립_오류", ""),
                        "error_code": d.get("error_code", ""), "justification_level": d.get("justification_level", ""),
                        "비고": "학생_답안 칸에 풀이 형태로 다시 써 넣을 것 (① … ② …). 라벨은 변환 후 다시 확인"})
    print(f"1-4 변환 틀 {len(want)}건 → {out}")


if __name__ == "__main__":
    main()
