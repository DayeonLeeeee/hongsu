# coding: utf-8
"""
실험 1-8: 손글씨 OCR 엔진 비교.

손글씨 사진과 정답 텍스트(사람이 타이핑)를 받아 엔진별로 OCR을 돌리고,
한글 줄 / 수식 줄 / 혼합 줄 인식률, 교정 위험, (선택) 채점 결과가 바뀐 비율을 계산한다.

  python experiments/ocr_compare.py \
      --samples answers/ocr_samples.csv \
      --engines gemini,mathpix,claude,azure \
      --grade \
      --out experiments/out/1-8_ocr.md

엔진 (키가 없는 엔진은 건너뛰고 표에 '키 없음'으로 남긴다):
  gemini   GEMINI_API_KEY                          서버 /ocr와 같은 프롬프트·후처리
  mathpix  MATHPIX_APP_ID, MATHPIX_APP_KEY         서버 /ocr와 같은 설정
  claude   ANTHROPIC_API_KEY (+ OCR_CLAUDE_MODEL)  같은 OCR 프롬프트를 Claude 비전으로
  azure    AZURE_VISION_ENDPOINT, AZURE_VISION_KEY Azure AI Vision Read (language=ko)

MyScript는 펜 획 좌표가 필요해서 사진으로는 비교할 수 없다 (태블릿 획 수집 화면이 생긴 뒤 추가).
여러 엔진 결과를 합치는 '대조 단계'도 아직 없으므로 이 스크립트는 엔진 단독 성능만 잰다.

samples CSV 열:
  sample_id     표본 ID
  image         사진 경로 (CSV 파일 기준 상대경로 가능)
  problem_id    문항 ID (--grade에서 사용, 예: INV-R-01)
  정답_텍스트    사람이 타이핑한 정답. 학생이 쓴 그대로(틀린 값도 그대로), 줄은 줄바꿈으로 구분
  교정검사       (선택) 학생이 일부러 틀리게 쓴 식. 엔진이 이 식을 고쳐버리면 '교정 발생'으로 센다
"""
import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

HANGUL = re.compile(r"[가-힣]")
MATHY = re.compile(r"[0-9=+\-−×÷^/√∘⁻()<>≤≥a-zA-Z]")


# ─────────────────────────────────────
# 비교 지표
# ─────────────────────────────────────

def norm(s: str) -> str:
    """비교용 정규화: 유니코드 통일, 공백 제거, 비슷한 기호 통일."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("−", "-").replace("·", "×").replace("*", "×")
    return re.sub(r"\s+", "", s)


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(truth: str, pred: str) -> float:
    t, p = norm(truth), norm(pred)
    return lev(t, p) / max(1, len(t))


def line_class(line: str) -> str:
    h, m = bool(HANGUL.search(line)), bool(MATHY.search(line))
    if h and m:
        return "혼합"
    return "한글" if h else "수식"


def split_lines(text: str) -> list:
    return [ln for ln in (text or "").splitlines() if ln.strip()]


# ─────────────────────────────────────
# 엔진
# ─────────────────────────────────────

def _steps_to_lines(data) -> list:
    from hongsu.utils import normalize_plain
    steps = (data or {}).get("steps") or []
    return [normalize_plain(str(s.get("text", ""))) for s in steps if isinstance(s, dict) and s.get("text")]


def run_gemini(b64, mime):
    from hongsu import config, ocr
    if not config.GEMINI_API_KEY:
        return None
    return _steps_to_lines(ocr.call_gemini_ocr(b64, mime))


def run_mathpix(b64, mime):
    from hongsu import config, ocr
    from hongsu.utils import normalize_plain
    if not (config.MATHPIX_APP_ID and config.MATHPIX_APP_KEY):
        return None
    return split_lines(normalize_plain(ocr.call_mathpix(b64, mime)))


def run_claude(b64, mime):
    from hongsu import clients, config
    import rv2_grader
    from prompts import ocr as p_ocr
    if not config.ANTHROPIC_API_KEY:
        return None
    model = os.environ.get("OCR_CLAUDE_MODEL", "claude-opus-5-5")
    kw = dict(model=model, max_tokens=4000, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
        {"type": "text", "text": p_ocr.build_solution_ocr_prompt()}]}])
    if rv2_grader.model_defaults(model)["temperature"] is not None:
        kw["temperature"] = 0.0
    resp = clients.get_claude().messages.create(**kw)
    parsed, _ = rv2_grader.extract_result(resp.content)
    return _steps_to_lines(parsed)


def run_azure(b64, mime):
    """Azure AI Vision Read 3.2 (비동기: 제출 → 결과 폴링). 실제 호출은 키 발급 후 확인 필요."""
    import requests
    endpoint, key = os.environ.get("AZURE_VISION_ENDPOINT", ""), os.environ.get("AZURE_VISION_KEY", "")
    if not (endpoint and key):
        return None
    url = endpoint.rstrip("/") + "/vision/v3.2/read/analyze?language=ko"
    r = requests.post(url, headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"},
                      data=base64.b64decode(b64), timeout=30)
    r.raise_for_status()
    op = r.headers["Operation-Location"]
    for _ in range(30):
        time.sleep(1)
        j = requests.get(op, headers={"Ocp-Apim-Subscription-Key": key}, timeout=30).json()
        if j.get("status") == "succeeded":
            return [ln["text"] for page in j["analyzeResult"]["readResults"] for ln in page.get("lines", [])]
        if j.get("status") == "failed":
            raise RuntimeError(f"Azure Read 실패: {j}")
    raise TimeoutError("Azure Read 30초 내 응답 없음")


ENGINES = {"gemini": run_gemini, "mathpix": run_mathpix, "claude": run_claude, "azure": run_azure}


# ─────────────────────────────────────
# 채점 영향 (--grade)
# ─────────────────────────────────────

def grade_signature(problem_id: str, text: str) -> dict:
    """같은 채점기로 돌려 비교할 값: 주 코드, 첫 오류 기준 번호, 결과 상태."""
    from hongsu import clients
    import rv2_grader
    rec = rv2_grader.diagnose(clients.get_claude(), problem_id, text)
    outcome, _ = rv2_grader.classify_outcome(rec)
    res = rec.get("result") or {}
    wrong = sorted((ln for ln in res.get("lines") or [] if ln.get("judgment") == "wrong"),
                   key=lambda ln: ln.get("no") or 0)
    return {"code": res.get("main_error_code"), "first_criterion": wrong[0].get("matched_criterion") if wrong else None,
            "outcome": outcome, "error": rec.get("error")}


# ─────────────────────────────────────
# 실행
# ─────────────────────────────────────

def load_samples(path):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        img = r["image"]
        r["_image_path"] = img if os.path.isabs(img) else os.path.join(base, img)
    return rows


def evaluate(truth_lines, pred_lines, keep_token):
    per_class = defaultdict(list)
    for i, t in enumerate(truth_lines):
        p = pred_lines[i] if i < len(pred_lines) else ""
        per_class[line_class(t)].append(cer(t, p))
    joined_t, joined_p = "\n".join(truth_lines), "\n".join(pred_lines)
    corrected = None
    if keep_token:
        corrected = norm(keep_token) not in norm(joined_p)
    return {"cer": cer(joined_t, joined_p), "per_class": dict(per_class),
            "n_truth_lines": len(truth_lines), "n_pred_lines": len(pred_lines), "corrected": corrected}


def summarize(records, engines, grade):
    lines = ["# 1-8 손글씨 OCR 엔진 비교", "",
             "CER(문자 오류율) = 틀린 글자 수 / 정답 글자 수. 낮을수록 좋다. 공백은 무시하고 비교.", "",
             "| 엔진 | 처리 | 전체 CER | 한글 줄 CER | 수식 줄 CER | 혼합 줄 CER | 줄 수 다름 | 교정 발생 | 평균 시간(초) |"
             + (" 채점 바뀜 |" if grade else ""),
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |" + (" --- |" if grade else "")]
    for eng in engines:
        rs = [r for r in records if r["engine"] == eng]
        ok = [r for r in rs if r.get("status") == "ok"]
        if not rs:
            continue
        if not ok:
            status = rs[0].get("status")
            lines.append(f"| {eng} | {status} |" + " |" * (8 if grade else 7))
            continue

        def mean(xs):
            return f"{sum(xs) / len(xs):.3f}" if xs else "-"
        cls = defaultdict(list)
        for r in ok:
            for k, v in r["eval"]["per_class"].items():
                cls[k].extend(v)
        corr = [r for r in ok if r["eval"]["corrected"] is not None]
        n_corr = sum(1 for r in corr if r["eval"]["corrected"])
        diff_lines = sum(1 for r in ok if r["eval"]["n_pred_lines"] != r["eval"]["n_truth_lines"])
        row = (f"| {eng} | {len(ok)}/{len(rs)} | {mean([r['eval']['cer'] for r in ok])} | {mean(cls['한글'])} | "
               f"{mean(cls['수식'])} | {mean(cls['혼합'])} | {diff_lines} | "
               f"{f'{n_corr}/{len(corr)}' if corr else '-'} | {mean([r['seconds'] for r in ok])[:4]} |")
        if grade:
            g = [r for r in ok if r.get("grade_changed") is not None]
            row += f" {sum(1 for r in g if r['grade_changed'])}/{len(g)} |"
        lines.append(row)
    lines += ["", "- 줄 수 다름: 엔진이 나눈 줄 수가 정답과 다른 표본 수. 줄이 밀리면 줄별 CER이 커진다.",
              "- 교정 발생: 교정검사 열의 틀린 식이 OCR 결과에서 사라진 표본 수 (학생 실수를 고쳐버림).",
              "- 채점 바뀜: 정답 텍스트로 채점한 결과와 주 코드·첫 오류 기준 번호·결과 상태 중 하나라도 다른 표본 수."]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--engines", default="gemini,mathpix,claude,azure")
    ap.add_argument("--grade", action="store_true", help="정답 텍스트와 OCR 결과를 각각 채점해 결과가 바뀌는지 본다 (API 비용 발생)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "experiments", "out", "1-8_ocr.md"))
    args = ap.parse_args(argv)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    bad = [e for e in engines if e not in ENGINES]
    if bad:
        raise SystemExit(f"모르는 엔진: {bad} (가능: {list(ENGINES)})")
    samples = load_samples(args.samples)
    raw_path = os.path.splitext(args.out)[0] + "_raw.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    records, truth_grade = [], {}
    with open(raw_path, "w", encoding="utf-8") as raw:
        for s in samples:
            with open(s["_image_path"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = mimetypes.guess_type(s["_image_path"])[0] or "image/jpeg"
            truth = split_lines(s["정답_텍스트"])
            if args.grade and s["sample_id"] not in truth_grade:
                truth_grade[s["sample_id"]] = grade_signature(s["problem_id"], "\n".join(truth))
            for eng in engines:
                rec = {"sample_id": s["sample_id"], "engine": eng}
                t0 = time.time()
                try:
                    pred = ENGINES[eng](b64, mime)
                    if pred is None:
                        rec["status"] = "키 없음"
                    else:
                        rec.update(status="ok", pred=pred, eval=evaluate(truth, pred, s.get("교정검사", "").strip()))
                except Exception as e:
                    rec.update(status="실패", error=f"{type(e).__name__}: {e}")
                rec["seconds"] = round(time.time() - t0, 2)
                if args.grade and rec["status"] == "ok":
                    sig = grade_signature(s["problem_id"], "\n".join(rec["pred"]))
                    ref = truth_grade[s["sample_id"]]
                    rec["grade"], rec["grade_truth"] = sig, ref
                    rec["grade_changed"] = any(sig[k] != ref[k] for k in ("code", "first_criterion", "outcome"))
                records.append(rec)
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw.flush()
                print(f"  {s['sample_id']} {eng}: {rec['status']}"
                      + (f" CER={rec['eval']['cer']:.3f}" if rec["status"] == "ok" else f" {rec.get('error', '')[:60]}"))

    text = summarize(records, engines, args.grade)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\n저장: {args.out}\n원자료: {raw_path}")


if __name__ == "__main__":
    main()
