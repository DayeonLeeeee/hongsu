# coding: utf-8
"""hongsu 서버 패키지.

  config.py         환경변수·상수
  logging_setup.py  로깅, 요청별 request_id
  clients.py        Claude·Gemini·Supabase 클라이언트
  utils.py          시각, 이미지 블록, 응답 파싱, 평문 정리
  ocr.py            Mathpix + Gemini OCR
  storage.py        인메모리 문항·결과 DB, Supabase 저장, 엑셀 문항 로드
  app.py            앱 조립 (create_app)
  routes/           엔드포인트 (ocr, grade, problems, results, system)

채점 로직은 저장소 루트의 rv2_grader.py, 엑셀 로더는 rv2_codebook.py, 프롬프트는 prompts/.
"""
