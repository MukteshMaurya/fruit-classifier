"""Security regression tests: no private keys/tokens in frontend
source or generated build output."""
import re
from pathlib import Path

import pytest

# Concrete credential formats that must never appear in files
# shipped to or built by the frontend.
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",           # OpenAI-style keys
    r"ghp_[A-Za-z0-9]{36,}",          # GitHub PATs
    r"gho_[A-Za-z0-9]{36,}",          # GitHub OAuth tokens
    r"ghu_[A-Za-z0-9]{36,}",          # GitHub user tokens
    r"hf_[A-Za-z0-9]{30,}",           # Hugging Face tokens
    r"AKIA[0-9A-Z]{16}",              # AWS access key ids
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",  # JWTs
]

SCANNABLE_SUFFIXES = {".js", ".html", ".css", ".json", ".txt"}


def _frontend_dir():
    # backend/tests/test_security.py -> repo root -> frontend
    return Path(__file__).resolve().parents[2] / "frontend"


def test_frontend_source_has_no_secrets():
    frontend = _frontend_dir()
    if not frontend.exists():
        pytest.skip("frontend/ is not deployed alongside the backend")
    scanned = 0
    for path in frontend.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCANNABLE_SUFFIXES:
            continue
        if "node_modules" in path.parts:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            assert re.search(pattern, content) is None, (
                f"possible secret ({pattern}) in {path}"
            )
        scanned += 1
    assert scanned > 0, "no frontend files found to scan"


def test_frontend_only_bakes_the_api_url():
    """build.js may only bake VITE_API_URL into the bundle —
    no other environment value may leak into generated config."""
    build_js = _frontend_dir() / "build.js"
    if not build_js.exists():
        pytest.skip("frontend/ is not deployed alongside the backend")
    content = build_js.read_text(encoding="utf-8")
    assert "process.env.VITE_API_URL" in content
    # every process.env reference in build.js must be VITE_API_URL
    env_refs = re.findall(r"process\.env\.([A-Z0-9_]+)", content)
    assert set(env_refs) == {"VITE_API_URL"}
