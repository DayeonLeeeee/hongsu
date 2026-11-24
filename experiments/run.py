# coding: utf-8
"""
실험 실행기 — Windows(PowerShell)와 macOS(zsh)에서 같은 명령으로 돌린다.

  python experiments/run.py setup --kit <rv2_kit 폴더>   처음 한 번 (키트 위치 기억 + 답안 파일 준비)
  python experiments/run.py status                     지금 할 수 있는 실험과 사람이 할 일 확인
  python experiments/run.py baseline                   1-1a·1-3·1-5·1-6·1-7·1-9 (11/11 결과, 비용 0)
  python experiments/run.py 1-2                        라벨 통일 후 첫 오류 줄 (비용 0)
  python experiments/run.py 1-1b                       1.3 코드로 expert·holdout 재채점 + 1.2와 비교 + 1-9 (API 비용)
  python experiments/run.py 1-4                        설명문형 5건 변환본 채점 + 비교 (API 비용)
  python experiments/run.py 1-8 [--grade]              손글씨 OCR 엔진 비교 (OCR API 비용)

macOS에서는 python 대신 python3를 쓴다. 보고서는 experiments/out/ 아래에 쌓인다.
API 비용이 드는 명령은 호출 수를 보여 주고 확인을 받는다 (--yes로 건너뜀).
"""
import argparse
import csv
import datetime
import glob
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

ANSWERS = os.path.join(ROOT, "answers")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "out")
KIT_PATH_FILE = os.path.join(ANSWERS, ".kit_path")

# 11/11 기준 결과 (키트 results/ 안). 프롬프트 1.2, Opus 5.5, 반복 3회
KIT_EXPERT_1111 = "stage3_expert_33_opus55_20261111_1422.jsonl"
KIT_HOLDOUT_1111 = "stage3_holdout45_opus55_20261111_1531.jsonl"

IS_WIN = platform.system() == "Windows"
PY = "python" if IS_WIN else "python3"


# ─────────────────────────────────────
# 공통
# ─────────────────────────────────────

def say(msg=""):
    print(msg, flush=True)


def stop(msg):
    say(f"\n[중단] {msg}")
    sys.exit(1)


def env_hint(name, example):
    return (f'PowerShell: $env:{name} = "{example}"' if IS_WIN else f'macOS: export {name}="{example}"')


def stamp():
    return datetime.datetime.now().strftime("%m%d_%H%M")


def kit_dir(arg_kit=""):
    """키트 위치: --kit → 환경변수 RV2_KIT → setup이 기억한 위치."""
    path = arg_kit or os.environ.get("RV2_KIT", "")
    if not path and os.path.exists(KIT_PATH_FILE):
        with open(KIT_PATH_FILE, encoding="utf-8") as f:
            path = f.read().strip()
    path = os.path.expanduser(path)
    if not path:
        stop("키트 위치를 모름. 먼저 실행: " + f'{PY} experiments/run.py setup --kit "<rv2_kit 폴더>"')
    return find_kit(path)


def find_kit(path):
    """키트 폴더 확인. zip을 '압축 풀기'하면 rv2_kit/rv2_kit 처럼 한 겹 더 생기므로 안쪽도 찾아본다."""
    if not os.path.isdir(path):
        stop(f"키트 폴더가 없음: {path}")
    def is_kit(p):   # hongsu 저장소에도 answers/expert_33.csv가 생기므로 키트에만 있는 파일로 구분
        return all(os.path.exists(os.path.join(p, m)) for m in (os.path.join("answers", "expert_33.csv"), "score_rv2.py"))
    if is_kit(path):
        return path
    inner = [os.path.join(path, d) for d in sorted(os.listdir(path)) if is_kit(os.path.join(path, d))]
    if len(inner) == 1:
        say(f"(키트가 한 겹 안쪽에 있음 → {inner[0]})")
        return inner[0]
    stop(f"키트 폴더가 아님 (answers/expert_33.csv 없음): {path}\n"
         "  → zip을 푼 뒤 answers, materials, results 폴더가 바로 보이는 폴더를 --kit으로 줄 것")


def kit_result(kit, name):
    p = os.path.join(kit, "results", name)
    if not os.path.exists(p):
        stop(f"키트에 11/11 결과 파일이 없음: {p}")
    return p


def need(path, how):
    if not os.path.exists(path):
        stop(f"파일 없음: {os.path.relpath(path, ROOT)}\n  → {how}")
    return path


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _kit_soft():
    """기억한 키트 위치 (없으면 None, 멈추지 않음)."""
    path = os.environ.get("RV2_KIT", "")
    if not path and os.path.exists(KIT_PATH_FILE):
        with open(KIT_PATH_FILE, encoding="utf-8") as f:
            path = f.read().strip()
    path = os.path.expanduser(path)
    return path if path and os.path.isdir(path) else None


def _write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def check_answer_csvs():
    """answers/의 CSV를 엑셀로 열어 저장했을 때 생기는 문제를 실험 전에 잡는다.
    - 한국어 Windows 엑셀의 일반 'CSV(쉼표로 분리)' 저장은 cp949라 f⁻¹의 '⁻' 같은 글자가 '?'로 바뀐다.
      → UTF-8(BOM)로 바꾸고 원본은 .cp949.bak으로 남긴다.
    - expert_33·holdout45는 키트에서 온 열(학생_답안, 라벨)을 키트 원본으로 되돌린다.
      사람이 고치는 열은 expert_33의 '전문가_첫오류_통일' 하나뿐이므로 그 열만 그대로 둔다.
    - 엑셀은 '(1)'을 회계 음수 -1로 바꾼다 → 첫 오류 라벨이 조용히 '없음'이 되므로 멈춘다.
    - 그 밖의 파일(converted_5 등)은 사람이 쓴 내용이라 되돌릴 원본이 없다 → '?'가 생긴 칸을 알려 준다."""
    kit = _kit_soft()
    paths = glob.glob(os.path.join(ANSWERS, "*.csv")) + glob.glob(os.path.join(ANSWERS, "ocr", "*.csv"))
    for path in paths:
        rel = os.path.relpath(path, ROOT)
        with open(path, "rb") as f:
            raw = f.read()
        was_cp949 = False
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp949")
            except UnicodeDecodeError:
                stop(f"{rel}: 글자 인코딩을 읽을 수 없음. 엑셀에서 'CSV UTF-8(쉼표로 분리)'로 다시 저장할 것")
            os.replace(path, path + ".cp949.bak")
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(text)
            was_cp949 = True
            say(f"(엑셀 일반 CSV(cp949) 저장 → UTF-8로 바꿈: {rel}, 원본은 .cp949.bak)")
        rows = read_csv(path)
        if not rows:
            continue
        fields = list(rows[0].keys())
        name = os.path.basename(path)
        kit_csv = os.path.join(kit, "answers", name) if kit else ""
        if name in ("expert_33.csv", "holdout45.csv") and kit_csv and os.path.exists(kit_csv):
            orig = {r["answer_id"]: r for r in read_csv(kit_csv)}
            fixed = set()
            for r in rows:
                o = orig.get(r.get("answer_id"))
                for k, v in (o or {}).items():
                    if k in r and r[k] != v:
                        r[k] = v
                        fixed.add(r["answer_id"])
            if fixed:
                _write_csv(path, rows, fields)
                say(f"({rel}: 키트에서 온 칸이 바뀌어 있어 키트 원본으로 되돌림 — {sorted(fixed)}. "
                    "'전문가_첫오류_통일' 열은 그대로)")
        elif was_cp949:
            q = [r.get("answer_id", "?") for r in rows if any("?" in (v or "") for v in r.values())]
            if q:
                say(f"주의: {rel}에서 '?'가 있는 행 {q} — 엑셀 저장 중 ⁻¹ 같은 글자가 '?'로 바뀌었을 수 있음. 확인할 것")
        bad = [f"{r.get('answer_id', '?')} {k}={v.strip()}" for r in rows for k, v in r.items()
               if k and "첫오류" in k and re.fullmatch(r"\s*-\d+\s*", v or "")]
        if bad:
            stop(f"{rel}: 엑셀이 '(1)' 같은 라벨을 음수로 바꾼 것으로 보임\n  "
                 + "\n  ".join(bad) + "\n  → 해당 칸을 숫자 하나(예: 1)로 고쳐 저장할 것 (메모장·VS Code로 열면 안전)")


def report(argv, name):
    """score_experiments 실행 → experiments/out/<name>.md"""
    import score_experiments
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{name}.md")
    say(f"\n── {name} ──")
    score_experiments.main(argv + ["--out", out])
    return out


def compare(argv, name):
    import compare_runs
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{name}.md")
    say(f"\n── {name} ──")
    compare_runs.main(argv + ["--out", out])
    return out


def confirm(calls, what, yes):
    say(f"\n{what}: Claude API 약 {calls}회 호출 (모델 {model_name()}, 비용 발생)")
    if yes:
        return
    try:
        ans = input("계속할까? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes", "ㅛ"):
        stop("취소함")


def model_name():
    import rv2_grader
    return rv2_grader.RV2_MODEL


def need_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        stop("ANTHROPIC_API_KEY가 없음.\n  " + env_hint("ANTHROPIC_API_KEY", "sk-ant-..."))


def run_batch(answers_csv, repeats=3):
    """run_batch.py를 돌리고 이번에 새로 생긴 결과 파일 경로를 돌려준다."""
    stem = os.path.splitext(os.path.basename(answers_csv))[0]
    before = set(glob.glob(os.path.join(RESULTS, f"stage3_{stem}_*.jsonl")))
    cmd = [sys.executable, os.path.join(ROOT, "run_batch.py"), "--answers", answers_csv, "--repeats", str(repeats)]
    say("\n$ " + " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c for c in cmd))
    env = dict(os.environ, PYTHONUTF8="1")
    if subprocess.call(cmd, cwd=ROOT, env=env) != 0:
        stop("run_batch.py 실패")
    new = sorted(set(glob.glob(os.path.join(RESULTS, f"stage3_{stem}_*.jsonl"))) - before, key=os.path.getmtime)
    if not new:
        stop("결과 파일을 찾지 못함")
    return new[-1]


def latest_result(stem):
    files = sorted(glob.glob(os.path.join(RESULTS, f"stage3_{stem}_*.jsonl")), key=os.path.getmtime)
    return files[-1] if files else None


# ─────────────────────────────────────
# 명령
# ─────────────────────────────────────

def cmd_setup(a):
    kit = os.path.expanduser(a.kit or os.environ.get("RV2_KIT", ""))
    if not kit:
        stop(f'키트 위치를 줘야 함: {PY} experiments/run.py setup --kit "<rv2_kit 폴더>"')
    kit = find_kit(kit)
    os.makedirs(ANSWERS, exist_ok=True)
    with open(KIT_PATH_FILE, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(kit))
    say(f"키트 위치 기억: {os.path.abspath(kit)}  (answers/.kit_path)")
    import prepare_answers
    prepare_answers.main(["--kit", kit, "--out-dir", ANSWERS] + (["--force"] if a.force else []))
    for name in (KIT_EXPERT_1111, KIT_HOLDOUT_1111):
        say(("  있음  " if os.path.exists(os.path.join(kit, "results", name)) else "  없음! ") + name)
    import importlib.util
    lacking = [m for m in ("anthropic", "openpyxl", "flask") if importlib.util.find_spec(m) is None]
    if lacking:
        say(f"\n패키지 부족: {', '.join(lacking)} → {PY} -m pip install -r requirements.txt")
    say(f"\n다음: {PY} experiments/run.py status")


def cmd_status(a):
    kit = kit_dir(a.kit)
    say(f"키트: {kit}")
    say(f"모델(새 채점 시): {model_name()}   API 키: {'있음' if os.environ.get('ANTHROPIC_API_KEY') else '없음'}")
    say("")

    def mark(ok):
        return "OK " if ok else "-- "

    base_ok = all(os.path.exists(os.path.join(kit, "results", n)) for n in (KIT_EXPERT_1111, KIT_HOLDOUT_1111))
    say(f"{mark(base_ok)}baseline (1-1a·1-3·1-5·1-6·1-7·1-9)  비용 0  → {PY} experiments/run.py baseline")

    exp = os.path.join(ANSWERS, "expert_33.csv")
    if os.path.exists(exp):
        rows = read_csv(exp)
        changed = [r["answer_id"] for r in rows if r.get("전문가_첫오류_통일", "") != r.get("전문가_첫오류", "")]
        say(f"{mark(bool(changed))}1-2 라벨 통일: 고친 답안 {len(changed)}건 {changed}")
        if changed:
            say(f"     → {PY} experiments/run.py 1-2")
        else:
            say("     할 일: answers/expert_33.csv의 '전문가_첫오류_통일'에서 A-01-A, A-01-B, A-04-A, A-04-B를 기준표 번호로")
    else:
        say(f"-- 1-2: answers/expert_33.csv 없음 → {PY} experiments/run.py setup")

    new_exp, new_hold = latest_result("expert_33"), latest_result("holdout45")
    say(f"{mark(bool(new_exp and new_hold))}1-1b 1.3 재채점: expert {'있음' if new_exp else '없음'}, holdout {'있음' if new_hold else '없음'}"
        f"  → {PY} experiments/run.py 1-1b")

    conv = os.path.join(ANSWERS, "converted_5.csv")
    if os.path.exists(conv):
        filled = [r["answer_id"] for r in read_csv(conv) if (r.get("학생_답안") or "").strip()]
        say(f"{mark(len(filled) == 5)}1-4 변환 답안: {len(filled)}/5 작성 {filled}")
    samples = os.path.join(ANSWERS, "ocr", "ocr_samples.csv")
    n = len(read_csv(samples)) if os.path.exists(samples) else 0
    say(f"{mark(n >= 30)}1-8 손글씨 표본: {n}장 (목표 30~50, answers/ocr/ocr_samples.csv)")
    say(f"\n보고서 폴더: {os.path.relpath(OUT, ROOT)}")


def cmd_baseline(a):
    kit = kit_dir(a.kit)
    exp = need(os.path.join(ANSWERS, "expert_33.csv"), f"{PY} experiments/run.py setup")
    hold = need(os.path.join(ANSWERS, "holdout45.csv"), f"{PY} experiments/run.py setup")
    s = stamp()
    outs = [report(["--results", kit_result(kit, KIT_EXPERT_1111), "--answers", exp], f"baseline_expert33_1111_{s}"),
            report(["--results", kit_result(kit, KIT_HOLDOUT_1111), "--answers", hold], f"baseline_holdout45_1111_{s}")]
    say("\n저장: " + ", ".join(os.path.relpath(o, ROOT) for o in outs))


def cmd_1_2(a):
    kit = kit_dir(a.kit)
    exp = need(os.path.join(ANSWERS, "expert_33.csv"), f"{PY} experiments/run.py setup")
    rows = read_csv(exp)
    if rows and "전문가_첫오류_통일" not in rows[0]:
        stop("answers/expert_33.csv에 '전문가_첫오류_통일' 열이 없음 → setup --force 로 다시 만들거나 열을 추가")
    changed = [r["answer_id"] for r in rows if r.get("전문가_첫오류_통일", "") != r.get("전문가_첫오류", "")]
    if not changed:
        say("주의: '전문가_첫오류_통일'이 원래 라벨과 같음 (아직 안 고침). 결과는 9/17과 같게 나온다.")
    else:
        say(f"통일 라벨로 고친 답안: {changed}")
    s = stamp()
    report(["--results", kit_result(kit, KIT_EXPERT_1111), "--answers", exp, "--first-col", "전문가_첫오류_통일",
            "--only", "1-2,focus"], f"1-2_unified_1111_{s}")
    newest = latest_result("expert_33")   # 1-1b를 돌렸으면 1.3 결과로도
    if newest:
        report(["--results", newest, "--answers", exp, "--first-col", "전문가_첫오류_통일", "--only", "1-2,focus"],
               f"1-2_unified_1.3_{s}")


def cmd_1_1b(a):
    kit = kit_dir(a.kit)
    exp = need(os.path.join(ANSWERS, "expert_33.csv"), f"{PY} experiments/run.py setup")
    hold = need(os.path.join(ANSWERS, "holdout45.csv"), f"{PY} experiments/run.py setup")
    if a.reuse:
        new_exp, new_hold = latest_result("expert_33"), latest_result("holdout45")
        if not (new_exp and new_hold):
            stop("--reuse: results/에 1.3 결과가 없음")
    else:
        need_api_key()
        confirm(33 * 3 + 40 * 3, "1-1b expert 33건 × 3회 + holdout 40건 × 3회 (형식 재시도가 있으면 조금 더)", a.yes)
        new_exp = run_batch(exp)
        new_hold = run_batch(hold)
    s = stamp()
    report(["--results", new_exp, "--answers", exp], f"1-1b_expert33_1.3_{s}")
    report(["--results", new_hold, "--answers", hold], f"1-1b_holdout45_1.3_{s}")
    for stem, base, new, ans in (("expert33", KIT_EXPERT_1111, new_exp, exp), ("holdout45", KIT_HOLDOUT_1111, new_hold, hold)):
        compare(["--ids", "all", "--title", f"1-1b {stem} 1.2(11/11) vs 1.3", "--before-tag", "1.2", "--after-tag", "1.3",
                 "--before", kit_result(kit, base), "--before-answers", ans, "--after", new, "--after-answers", ans],
                f"1-1b_{stem}_compare_{s}")
    say("\n1-9(코드 개선 후 재확인)는 위 1.3 보고서의 '지정 답안 확인' 섹션. 11/11 기준은 baseline 보고서와 비교.")


def cmd_1_4(a):
    kit = kit_dir(a.kit)
    exp = need(os.path.join(ANSWERS, "expert_33.csv"), f"{PY} experiments/run.py setup")
    conv = need(os.path.join(ANSWERS, "converted_5.csv"), f"{PY} experiments/run.py setup")
    filled = [r["answer_id"] for r in read_csv(conv) if (r.get("학생_답안") or "").strip()]
    if not filled:
        stop("answers/converted_5.csv의 '학생_답안'이 전부 비어 있음. 풀이 형태로 먼저 채울 것")
    if len(filled) < 5:
        say(f"주의: {len(filled)}/5건만 채워짐 {filled}. 빈 답안은 건너뛴다.")
    if a.reuse:
        new = latest_result("converted_5") or stop("--reuse: results/에 converted_5 결과가 없음")
    else:
        need_api_key()
        confirm(len(filled) * 3, f"1-4 변환 답안 {len(filled)}건 × 3회", a.yes)
        new = run_batch(conv)
    before = latest_result("expert_33") if a.before_13 else kit_result(kit, KIT_EXPERT_1111)
    if a.before_13 and not before:
        stop("--before-13: results/에 1.3 expert 결과가 없음 (먼저 1-1b)")
    compare(["--before", before, "--before-answers", exp, "--after", new, "--after-answers", conv],
            f"1-4_converted_{stamp()}")
    if not a.before_13:
        say("\n(변환 전 = 11/11 결과(1.2), 변환 후 = 1.3 코드. 1-1b에서 판정 차이가 있었으면 --before-13 으로 다시 비교)")


def cmd_1_8(a):
    samples = need(os.path.join(ANSWERS, "ocr", "ocr_samples.csv"),
                   "experiments/templates/ocr_samples.csv를 answers/ocr/ocr_samples.csv로 복사해서 채우고, 사진은 answers/ocr/images/에")
    n = len(read_csv(samples))
    missing = [k for k in ("GEMINI_API_KEY", "MATHPIX_APP_ID", "AZURE_VISION_KEY") if not os.environ.get(k)]
    if missing:
        say("키가 없어 건너뛸 엔진: " + ", ".join(missing) + "\n  예) " + env_hint("GEMINI_API_KEY", "..."))
    if a.grade:
        need_api_key()
        confirm(n * 5, f"1-8 채점 영향 (표본 {n}개 × (엔진 4 + 정답 1))", a.yes)
    import ocr_compare
    os.makedirs(OUT, exist_ok=True)
    argv = ["--samples", samples, "--engines", a.engines, "--out", os.path.join(OUT, f"1-8_ocr_{stamp()}.md")]
    ocr_compare.main(argv + (["--grade"] if a.grade else []))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "status", "baseline", "1-2", "1-1b", "1-4", "1-8"])
    ap.add_argument("--kit", default="", help="rv2_kit 폴더 (setup 후에는 생략 가능)")
    ap.add_argument("--yes", action="store_true", help="API 비용 확인 질문을 건너뜀")
    ap.add_argument("--force", action="store_true", help="setup: answers/ 파일을 새로 만듦 (고친 라벨이 지워짐)")
    ap.add_argument("--reuse", action="store_true", help="1-1b·1-4: 새로 채점하지 않고 results/의 최신 결과로 분석만")
    ap.add_argument("--before-13", action="store_true", help="1-4: 변환 전 비교 대상을 1.3 expert 결과로")
    ap.add_argument("--grade", action="store_true", help="1-8: 채점 영향까지 (채점 비용)")
    ap.add_argument("--engines", default="gemini,mathpix,claude,azure", help="1-8: 비교할 OCR 엔진")
    a = ap.parse_args(argv)
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):   # 출력을 파일로 돌려도 한글이 깨지지 않게
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if a.command != "setup":
        check_answer_csvs()
    {"setup": cmd_setup, "status": cmd_status, "baseline": cmd_baseline, "1-2": cmd_1_2,
     "1-1b": cmd_1_1b, "1-4": cmd_1_4, "1-8": cmd_1_8}[a.command](a)


if __name__ == "__main__":
    main()
