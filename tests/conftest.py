"""Shared pytest fixtures.

We force ``LLM_PROVIDER=none`` for the test suite so it never hits a real API,
regardless of which env vars are present on the developer's machine.
"""

from __future__ import annotations

import os

import pytest

os.environ["LLM_PROVIDER"] = "none"
# Defensively scrub keys so a leaked env var can't trigger a network call.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture()
def client():
    from app.api import create_app

    flask_app = create_app()
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client
