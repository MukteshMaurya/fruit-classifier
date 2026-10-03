from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def samples_dir():
    return SAMPLES_DIR
