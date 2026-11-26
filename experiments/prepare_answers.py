# coding: utf-8
"""
실험용 답안 파일 준비 (키트 라벨 형식).

  python experiments/prepare_answers.py --kit "C:/Users/USER/Desktop/연구/rv2_kit"

만드는 파일 (answers/ 아래, 이미 있으면 덮어쓰지 않는다 — 사람이 고친 라벨 보호. 새로 만들려면 --force):
  expert_33.csv      키트 answers/expert_33.csv 복사. 키트가 없으면 엑셀 04 시트에서 같은 열 이름으로 만든다.
                     라벨 통일용 '전문가_첫오류_통일'(1-2), '전문가_코드_통일'(미완 처리, '미완' = 코드 없음) 열을 덧붙인다
                     (처음엔 원래 라벨 복사. 사람이 통일 대상만 고친다).
                     첫 오류 두 열 '전문가_첫오류_학생줄'(학생 답안 줄 번호), '전문가_오류기준'(어긴 기준표 번호)은
                     빈칸으로 덧붙인다. 기준표만 보고 사람이 채운다(--merge-first-error 시트.xlsx 로 옮길 수 있음).
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
UNIFIED_CODE_COL = "전문가_코드_통일"   # 주 코드 통일본 ('미완' = 코드 없음 + 미완, 예: A-06-A)
# 첫 오류를 두 열로 (11/11 합의 2와 짝). 사람이 기준표만 보고 채운다 — 모델 결과를 보고 채우지 않는다.
FIRST_LINE_COL = "전문가_첫오류_학생줄"   # 학생 답안의 몇 번째 줄에서 처음 틀렸나 (정수)
ERR_CRIT_COL = "전문가_오류기준"          # 그 오류가 어긴 기준표 번호 (정수)


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


def add_unified_cols(path):
    """라벨 통일 열이 없으면 원래 라벨을 복사해 덧붙인다 (사람이 통일 대상만 고친다). 고친 열이 있으면 건드리지 않는다."""
    rows = read_csv(path)
    if not rows:
        return []
    added = []
    for col, src in ((UNIFIED_COL, "전문가_첫오류"), (UNIFIED_CODE_COL, "전문가_코드")):
        if col not in rows[0]:
            for r in rows:
                r[col] = r.get(src, "")
            added.append(col)
    if added:
        _write(path, list(rows[0].keys()), rows)
        print(f"  {', '.join(repr(c) for c in added)} 열 추가 (원래 라벨 복사본 — 통일 대상만 고칠 것) : {path}")
    return added


def add_first_error_cols(path):
    """첫 오류 두 열이 없으면 빈칸으로 덧붙인다 (원래 라벨을 복사하지 않음 — 사람이 채운다)."""
    rows = read_csv(path)
    if not rows:
        return []
    added = [c for c in (FIRST_LINE_COL, ERR_CRIT_COL) if c not in rows[0]]
    if added:
        for r in rows:
            for c in added:
                r[c] = ""
        _write(path, list(rows[0].keys()), rows)
        print(f"  {', '.join(repr(c) for c in added)} 열 추가 (빈칸 — 기준표만 보고 사람이 채울 것) : {path}")
    return added


def merge_first_error(xlsx, csv_path):
    """사람이 채운 첫 오류 시트(answer_id, 전문가_첫오류_학생줄, 전문가_오류기준)를 답안 CSV에 옮긴다.
    시트에 값이 있는 칸만 옮기고, 정수가 아니면 멈춘다."""
    import openpyxl
    add_first_error_cols(csv_path)
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    head = [str(h).strip() if h is not None else "" for h in next(it)]
    need = ("answer_id", FIRST_LINE_COL, ERR_CRIT_COL)
    miss = [c for c in need if c not in head]
    if miss:
        raise SystemExit(f"시트에 열이 없음: {miss}")
    filled = {}
    for r in it:
        d = dict(zip(head, r))
        aid = str(d.get("answer_id") or "").strip()
        if not aid:
            continue
        for c in (FIRST_LINE_COL, ERR_CRIT_COL):
            v = d.get(c)
            if v is None or str(v).strip() == "":
                continue
            try:
                n = int(float(str(v).strip()))
            except ValueError:
                raise SystemExit(f"{aid} {c}: 정수가 아님 {v!r}")
            filled.setdefault(aid, {})[c] = str(n)
    rows = read_csv(csv_path)
    ids = {r["answer_id"] for r in rows}
    unknown = sorted(set(filled) - ids)
    if unknown:
        raise SystemExit(f"답안 CSV에 없는 answer_id: {unknown}")
    for r in rows:
        r.update(filled.get(r["answer_id"], {}))
    _write(csv_path, list(rows[0].keys()), rows)
    print(f"  첫 오류 두 열: {len(filled)}건 옮김 → {csv_path}")
    return filled


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
    ap.add_argument("--merge-first-error", default="",
                    help="사람이 채운 첫 오류 시트(xlsx)를 answers/expert_33.csv의 두 열에 옮기고 끝낸다")
    args = ap.parse_args(argv)

    if args.merge_first_error:
        merge_first_error(args.merge_first_error, os.path.join(args.out_dir, "expert_33.csv"))
        return

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
        add_unified_cols(expert_path)
    add_first_error_cols(expert_path)

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
            row[UNIFIED_CODE_COL] = e.get(UNIFIED_CODE_COL, e.get("전문가_코드", ""))
            row["학생_답안"] = ""
            row["비고"] = ((e.get("비고") or "") + " | 학생_답안 칸에 풀이 형태로 다시 써 넣을 것 (① … ② …). "
                         "라벨은 변환 후 다시 확인").strip(" |")
            rows.append(row)
        cols = KIT_COLS[:3] + ["원래_답안"] + KIT_COLS[3:] + [UNIFIED_CODE_COL]
        _write(conv_path, cols, rows)
        print(f"1-4 변환 틀 {len(rows)}건: {conv_path}")


if __name__ == "__main__":
    main()
