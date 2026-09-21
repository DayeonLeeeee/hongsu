"""
Hongsu 프롬프트 모음.

프롬프트는 코드와 분리해서 여기 모아둠. 각 파일의 VERSION 상수는
결과에 함께 저장되어 재현성 확보 용도.

파일 구성:
  - ocr.py           : Gemini OCR 프롬프트 (문제/풀이)
  - refine.py        : Haiku 자연어 수정 프롬프트
  - grading.py       : Opus 채점 프롬프트 (감점형)
  - classify.py      : Opus H코드 직접 분류 프롬프트
  - feedback.py      : Haiku 개인화 피드백 프롬프트

프롬프트 수정 시 각 파일 상단 VERSION을 반드시 갱신 (YYYY-MM-DD 포함).
"""

# 전체 프롬프트 버전 (개별 파일 버전과 별개, 서버 결과에 기록)
OVERALL_VERSION = "v2.2-2026-08-29"
