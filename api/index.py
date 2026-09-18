"""
api/index.py — Vercel ASGI entrypoint.

Vercel's Python runtime expects the ASGI app object named `app` in this file.
"""
from app.main import app  # noqa: F401
