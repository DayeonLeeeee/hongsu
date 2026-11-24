# coding: utf-8
"""
실험용 답안 파일 준비 (키트 라벨 형식).

  python experiments/prepare_answers.py --kit "C:/Users/USER/Desktop/연구/rv2_kit"

만드는 파일 (answers/ 아래, 이미 있으면 덮어쓰지 않는다 — 사람이 고친 라벨 보호. 새로 만들려면 --force):
  expert_33.csv      키트 answers/expert_33.csv 복사. 키트가 없으면 엑셀 04 시트에서 같은 열 이름으로 만든다.
                     라벨 통일(1-2)용 '전문가_첫오류_통일' 열을 덧붙인다 (처음엔 전문가_첫오류 복사. 사람이 고친다).
  holdout45.csv      키트 answers/holdout45.csv 복사 (--kit을 줬을 때만). 저장소에는 올라가지 않는다(.gitignore).
  converted_5.csv    1-4용 틀. 설명문형 5건의 원래 답안을 '원래_답안'에 두고 '학생_답안'은 비워 둔다.

열 이름은 키트 형식: answer_id, problem_id, 유형, 학생_답안, 전문가_줄별판정, 전문가_첫오류, 전문가_코드, 전문가_J, 비고
"""
import argparse
import csv
import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_COLS = ["answer_id", "problem_id", "유형", "학생_답안", "전문가_줄별판정", "전문가_첫오류", "전문가_코드", "전문가_J", "비고"]
XLSX_TO_KIT = {"줄별_판정": "전문가_줄별판정", "첫_독립_오류": "전문가_첫오류", "error_code": "전문가_코드",
               "justification_level": "전문가_J"}
CONVERTED_IDS = "A-05-D,A-06-A,A-06-C,A-08-A,A-09-B"
UNIFIED_COL = "전문가_첫오류_통일"


def _write(path, cols, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def expert_from_xlsx(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb["04_SYNTHETIC_ANSWERS"].iter_rows(values_only=True))
    head = [XLSX_TO_KIT.get(str(h).strip(), str(h).strip()) if h is not None else "" for h in rows[0]]
    out = []
    for r in rows[1:]:
        if r and r[0]:
            d = {h: ("" if v is None else str(v)) for h, v in zip(head, r)}
            out.append({c: d.get(c, "") for c in KIT_COLS})
    return out


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default="", help="rv2_kit 폴더 (있으면 키트 답안 파일을 그대로 복사)")
    ap.add_argument("--xlsx", default=os.path.join(BASE_DIR, "materials", "dataset_rv2.xlsx"))
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "answers"))
    ap.add_argument("--converted-ids", default=CONVERTED_IDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    def skip(path):
        if os.path.exists(path) and not args.force:
            print(f"이미 있음 (그대로 둠): {path}")
            return True
        return False

    expert_path = os.path.join(args.out_dir, "expert_33.csv")
    kit_expert = os.path.join(args.kit, "answers", "expert_33.csv") if args.kit else ""
    if not skip(expert_path):
        if kit_expert and os.path.exists(kit_expert):
            os.makedirs(args.out_dir, exist_ok=True)
            shutil.copyfile(kit_expert, expert_path)
            print(f"키트에서 복사: {expert_path}")
        else:
            _write(expert_path, KIT_COLS, expert_from_xlsx(args.xlsx))
            print(f"엑셀 04 시트에서 생성: {expert_path}")
        rows = read_csv(expert_path)
        if rows and UNIFIED_COL not in rows[0]:
            for r in rows:
                r[UNIFIED_COL] = r.get("전문가_첫오류", "")
            _write(expert_path, list(rows[0].keys()), rows)
            print(f"  '{UNIFIED_COL}' 열 추가 (전문가_첫오류 복사본 — 라벨 통일 대상만 고칠 것)")

    if args.kit:
        src = os.path.join(args.kit, "answers", "holdout45.csv")
        dst = os.path.join(args.out_dir, "holdout45.csv")
        if os.path.exists(src) and not skip(dst):
            shutil.copyfile(src, dst)
            print(f"키트에서 복사: {dst} (저장소에는 올라가지 않음)")

    conv_path = os.path.join(args.out_dir, "converted_5.csv")
    if not skip(conv_path):
        experts = {r["answer_id"]: r for r in read_csv(expert_path)}
        want = [x.strip() for x in args.converted_ids.split(",") if x.strip()]
        rows = []
        for aid in want:
            e = experts.get(aid, {"answer_id": aid})
            row = {c: e.get(c, "") for c in KIT_COLS}
            row["원래_답안"] = e.get("학생_답안", "")
            row["학생_답안"] = ""
            row["비고"] = ((e.get("비고") or "") + " | 학생_답안 칸에 풀이 형태로 다시 써 넣을 것 (① … ② …). "
                         "라벨은 변환 후 다시 확인").strip(" |")
            rows.append(row)
        cols = KIT_COLS[:3] + ["원래_답안"] + KIT_COLS[3:]
        _write(conv_path, cols, rows)
        print(f"1-4 변환 틀 {len(rows)}건: {conv_path}")


if __name__ == "__main__":
    main()
