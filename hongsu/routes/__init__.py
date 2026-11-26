# coding: utf-8
"""엔드포인트 묶음(Blueprint). URL은 분리 전과 같다."""
from flask import Flask

from hongsu.routes import grade, ocr, problems, results, system


def register(app: Flask) -> None:
    for module in (ocr, grade, problems, results, system):
        app.register_blueprint(module.bp)
