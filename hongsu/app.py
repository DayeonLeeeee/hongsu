# coding: utf-8
"""Flask 앱 조립: 로깅 훅, 에러 처리, 엔드포인트 등록, 초기 문항 로드."""
from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from hongsu import logging_setup, routes, storage
from hongsu.logging_setup import current_request_id, log


def _unhandled(e: Exception) -> ResponseReturnValue:
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    log.exception("[처리되지 않은 에러] %s %s", request.method, request.path)
    return jsonify({"success": False, "error": f"{type(e).__name__}: {e}",
                    "request_id": current_request_id()}), 500


def create_app() -> Flask:
    app = Flask(__name__)
    logging_setup.init_app(app)
    app.register_error_handler(Exception, _unhandled)
    routes.register(app)
    storage.load_initial_problems()
    return app
