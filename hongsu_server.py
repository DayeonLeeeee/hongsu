"""
hongsu 서버 (R v2)

파이프라인:
  1. 풀이 OCR:     Mathpix + Gemini Flash 병렬 → 단계별 평문 (/ocr)
  2. 자연어 수정:  Claude Haiku (vision + tool_use) (/refine) → 학생이 OCR 결과 확인
  3. R v2 진단:    확인된 OCR 텍스트 → 줄별 ok/wrong/depends + R코드 + J수준 (/grade)
                  키트(kit-1.2-20261022) 로직 이식. 문항·코드북·기준판정·규칙은 전부
                   엑셀(materials/dataset_rv2.xlsx)에서 조립, 문항별 허용 코드만 노출,
                   인용 원문 대조·허용 밖 코드 거부·1회 재시도, 기권(review_needed),
                   응답 원문·재현성 필드 저장 (Supabase rv2_gradings)
  4. 피드백:       다음 단계 (엑셀 08시트 질문→힌트→해설). 현재는 빈 값
  5. 모범답안      (/master_solution)

hongsu-rv2-1.1: 기존 H코드 체계(H1~H10 상수·분류·H 피드백·100점 루브릭 채점, /grade_h) 제거.
                문항은 엑셀 01_PROBLEMS(INV-R-01~10)만 사용.
hongsu-rv2-1.2: 코드북 R-v3 대응 — 진단 결과에 response_complete(요구 충족 여부)·
                missing_criteria(대응 풀이가 없는 기준 번호) 추가, Supabase 저장 포함.
                Supabase: sql/rv2_gradings_v12.sql 먼저 실행할 것.
hongsu-rv2-1.3: 안정화(응답 정제·재시도·로깅·상태 점검)와 구조 분리.
                코드는 hongsu/ 패키지로 옮겼고, 이 파일은 진입점만 남긴다
                (gunicorn hongsu_server:app). 구조는 hongsu/__init__.py 참고.
                Supabase: sql/2026-11-24_rv2_gradings_add_columns.sql 실행할 것.
"""
import os

from hongsu.app import create_app

app = create_app()


# ─── 서버 시작 (로컬 실행용) ────
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("=" * 60)
    print("  hongsu v2 서버 (vision + tool_use)")
    print("  로컬: http://localhost:5000")
    print(f"  네트워크: http://{local_ip}:5000")
    print(f"  학생 앱: http://{local_ip}:5000/student")
    print(f"  관리자:  http://{local_ip}:5000/admin")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
