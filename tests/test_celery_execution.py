import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.celery_app import validate_rolling_backend


@pytest.mark.parametrize(
    "backend", ["redis://localhost/0", "rediss://localhost/0", "redis+socket:///tmp/redis.sock"]
)
def test_rolling_backend_accepts_redis(backend):
    validate_rolling_backend(
        SimpleNamespace(conf=SimpleNamespace(result_backend=backend, task_ignore_result=False))
    )


@pytest.mark.parametrize(
    "backend", ["", "rpc://", "cache+memory://", "disabled://", "amqp://user:secret@host//"]
)
def test_rolling_backend_rejects_unsupported_without_leaking_url(backend):
    with pytest.raises(ValueError, match="Rolling chords require a Redis result backend") as exc:
        validate_rolling_backend(
            SimpleNamespace(conf=SimpleNamespace(result_backend=backend, task_ignore_result=False))
        )
    assert "secret" not in str(exc.value)


def test_worker_loads_rolling_tasks_in_fresh_process():
    script = """
import sys
from app.celery_app import celery_app
assert "app.tasks.rolling" not in sys.modules
celery_app.loader.import_default_modules()
expected = {"fetch_month_snapshot", "merge_rolling_12m", "train_rolling_12m",
            "gate_rolling_12m", "publish_rolling_12m", "trigger_rolling_12m"}
assert {"app.tasks.rolling." + name for name in expected} <= set(celery_app.tasks)
assert not celery_app.conf.task_ignore_result
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_worker_validates_only_training_queue(monkeypatch):
    from app.celery_app import ValidateRollingBackend

    validate = Mock(side_effect=ValueError("invalid training backend"))
    monkeypatch.setattr("app.celery_app.validate_rolling_backend", validate)
    worker = SimpleNamespace(
        app=SimpleNamespace(
            amqp=SimpleNamespace(queues=SimpleNamespace(consume_from={"celery": object()}))
        )
    )
    ValidateRollingBackend(worker)
    validate.assert_not_called()
    worker.app.amqp.queues.consume_from["training"] = object()
    with pytest.raises(ValueError, match="invalid training backend"):
        ValidateRollingBackend(worker)
    validate.assert_called_once_with(worker.app)


def test_trigger_rejects_backend_before_dispatch(monkeypatch):
    from app.tasks import rolling

    monkeypatch.setitem(rolling.trigger_rolling_12m.app.conf, "result_backend", "rpc://")
    dispatch = Mock()
    monkeypatch.setattr(rolling, "chord", dispatch)
    with pytest.raises(ValueError, match="Rolling chords require"):
        rolling.trigger_rolling_12m.run()
    dispatch.assert_not_called()


def test_cli_rejects_backend_before_enqueue(monkeypatch):
    from scripts import rolling_12m

    monkeypatch.setitem(rolling_12m.trigger_rolling_12m.app.conf, "result_backend", "rpc://")
    enqueue = Mock()
    monkeypatch.setattr(rolling_12m.trigger_rolling_12m, "delay", enqueue)
    monkeypatch.setattr(sys, "argv", ["rolling_12m"])
    with pytest.raises(ValueError, match="Rolling chords require"):
        rolling_12m.main()
    enqueue.assert_not_called()


@pytest.mark.parametrize(
    "ree_backend,override",
    [
        ("rpc://", ""),
        ("redis://localhost/0", "rpc://"),
        ("rpc://", "redis://localhost/0"),
    ],
)
def test_inference_import_does_not_validate_training_backend(ree_backend, override):
    env = dict(os.environ, REE_CELERY_RESULT_BACKEND=ree_backend, CELERY_RESULT_BACKEND=override)
    subprocess.run(
        [sys.executable, "-c", "from app.main import app; assert app"],
        env=env,
        check=True,
        timeout=30,
    )


def test_training_worker_bootstrap_rejects_rpc_without_services():
    env = dict(os.environ, REE_CELERY_RESULT_BACKEND="rpc://")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from app.celery_app import celery_app
try:
    celery_app.Worker(queues=["training"], pool="solo", concurrency=1)
except ValueError as exc:
    assert "Rolling chords require" in str(exc)
else:
    raise AssertionError("training worker accepted RPC backend")
""",
        ],
        env=env,
        check=True,
        timeout=30,
    )


def test_rolling_backend_requires_results():
    app = SimpleNamespace(
        conf=SimpleNamespace(
            result_backend="redis://user:secret@localhost/0",
            task_ignore_result=True,
        )
    )
    with pytest.raises(ValueError, match="task_ignore_result=False") as exc:
        validate_rolling_backend(app)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("entrypoint", ["worker", "cli", "trigger"])
@pytest.mark.parametrize(
    "ree_backend,override,accepted",
    [
        ("redis://localhost/0", "rpc://user:synthetic-secret@localhost//", False),
        ("rpc://", "redis://localhost/0", True),
        ("", "redis://localhost/0", True),
        ("rpc://", "rediss://localhost/0", True),
        ("rpc://", "redis+socket:///tmp/synthetic-redis.sock", True),
    ],
)
def test_effective_backend_overrides_in_fresh_process(entrypoint, ree_backend, override, accepted):
    env = dict(os.environ, REE_CELERY_RESULT_BACKEND=ree_backend, CELERY_RESULT_BACKEND=override)
    script = """
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

def no_network(*args, **kwargs):
    raise AssertionError("configuration validation attempted a connection")
socket.socket.connect = no_network
socket.getaddrinfo = no_network

from app.celery_app import celery_app
from app.config import settings
from app.tasks import rolling
from scripts import rolling_12m

assert settings.celery_result_backend == os.environ["REE_CELERY_RESULT_BACKEND"]
assert celery_app.conf.result_backend == os.environ["CELERY_RESULT_BACKEND"]
assert celery_app.conf.task_ignore_result is False
for name in ("fetch_month_snapshot", "merge_rolling_12m", "train_rolling_12m",
             "gate_rolling_12m", "publish_rolling_12m", "trigger_rolling_12m"):
    assert getattr(rolling, name).ignore_result is False
entrypoint, accepted = sys.argv[1], sys.argv[2] == "True"
dispatch = Mock(return_value=SimpleNamespace(id="synthetic-result"))
rolling.chord = Mock(return_value=dispatch)
rolling_12m.trigger_rolling_12m.delay = dispatch
try:
    if entrypoint == "worker":
        celery_app.Worker(queues=["training"], pool="solo", concurrency=1)
    elif entrypoint == "cli":
        sys.argv = ["rolling_12m"]
        rolling_12m.main()
    else:
        rolling.trigger_rolling_12m.run(as_of="2026-01-01", months=1)
except ValueError as exc:
    assert not accepted
    assert "Rolling chords require a Redis result backend" in str(exc)
    assert "synthetic-secret" not in str(exc)
    assert os.environ["CELERY_RESULT_BACKEND"] not in str(exc)
    dispatch.assert_not_called()
    rolling.chord.assert_not_called()
else:
    assert accepted
    if entrypoint != "worker":
        dispatch.assert_called_once()
"""
    subprocess.run(
        [sys.executable, "-c", script, entrypoint, str(accepted)], env=env, check=True, timeout=30
    )
