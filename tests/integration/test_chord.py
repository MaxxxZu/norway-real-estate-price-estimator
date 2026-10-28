import os
import subprocess
import sys
import uuid
from urllib.parse import urlsplit

import pytest
from celery import chord

pytestmark = pytest.mark.integration


def test_real_rabbitmq_redis_chord(tmp_path):
    # Only explicit, disposable loopback services; never inherit personal URLs.
    broker = os.environ.get("TEST_CELERY_BROKER_URL", "amqp://guest:guest@127.0.0.1:5672//")
    backend = os.environ.get("TEST_CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/15")
    for url, schemes in [(broker, {"amqp"}), (backend, {"redis"})]:
        parsed = urlsplit(url)
        assert parsed.scheme in schemes and parsed.hostname in {"localhost", "127.0.0.1"}
    from app.celery_app import validate_rolling_backend
    from tests.integration.chord_app import celery_app, echo, total

    queue = f"chord-test-{uuid.uuid4().hex}"
    celery_app.conf.update(broker_url=broker, result_backend=backend, task_always_eager=False)
    validate_rolling_backend(celery_app)
    env = dict(os.environ, REE_CELERY_BROKER_URL=broker, REE_CELERY_RESULT_BACKEND=backend)
    with (tmp_path / "worker.log").open("w+") as log:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "tests.integration.chord_app:celery_app",
                "worker",
                "--pool=solo",
                "--concurrency=1",
                "--loglevel=INFO",
                "--without-gossip",
                "--without-mingle",
                "--without-heartbeat",
                "-Q",
                queue,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            result = chord([echo.s(n).set(queue=queue) for n in [3, 1, 2]])(
                total.s().set(queue=queue)
            )
            assert result.get(timeout=45) == {"values": [1, 2, 3], "total": 6}
            assert worker.poll() is None
            result.forget()
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            with celery_app.connection_for_write() as connection:
                connection.default_channel.queue_delete(queue=queue)
