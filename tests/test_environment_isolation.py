import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import settings


def test_settings_are_synthetic_before_test_imports():
    assert settings.env == "test"
    assert settings.api_key == "synthetic-test-key"
    assert settings.api_base_url == "http://data-api.invalid"
    assert settings.s3_access_key == "synthetic-access-key"
    assert settings.s3_secret_key == "synthetic-secret-key"
    assert os.environ["AWS_SHARED_CREDENTIALS_FILE"] == os.devnull
    assert os.environ["AWS_EC2_METADATA_DISABLED"] == "true"
    assert "AWS_ACCESS_KEY_ID" not in os.environ
    assert "API_KEY" not in os.environ
    assert settings.train_min_rows == 500


def test_dotenv_is_disabled_even_in_another_working_directory(tmp_path):
    (tmp_path / ".env").write_text("REE_API_KEY=dotenv-sentinel\nREE_TRAIN_MIN_ROWS=9999\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from app.config import settings
assert settings.api_key == "synthetic-test-key"
assert settings.train_min_rows == 500
""",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=30,
    )


def test_offline_socket_guard(pytestconfig):
    if pytestconfig.getoption("--run-integration"):
        pytest.skip("network explicitly enabled")
    with socket.socket() as sock, pytest.raises(RuntimeError, match="Network disabled"):
        sock.connect(("127.0.0.1", 1))


def test_offline_fit_guard(pytestconfig):
    if pytestconfig.getoption("--run-training"):
        pytest.skip("fitting explicitly enabled")
    from catboost import CatBoostRegressor

    with pytest.raises(RuntimeError, match="Model fitting disabled"):
        CatBoostRegressor().fit([[1], [2]], [1, 2])


def test_pytest_sanitizes_inherited_environment(tmp_path):
    env = dict(
        os.environ,
        REE_API_KEY="inherited-sentinel",
        REE_TRAIN_MIN_ROWS="9999",
        AWS_ACCESS_KEY_ID="inherited-sentinel",
        API_KEY="inherited-sentinel",
    )
    env_file = tmp_path / "personal.env"
    env_file.write_text("REE_API_KEY=dotenv-sentinel\n")
    env["REE_ENV_FILE"] = str(env_file)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_environment_isolation.py",
            "-k",
            "settings_are_synthetic_before_test_imports",
            "-q",
        ],
        env=env,
        check=True,
        timeout=30,
    )
