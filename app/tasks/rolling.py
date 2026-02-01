import time
from datetime import date
from io import BytesIO

import joblib
from celery import chord

from app.celery_app import TrainingLongTask, TrainingShortTask, TrainingTask, celery_app
from app.clients.api_client import ApiClient
from app.config import settings
from app.observability.logging import log
from app.observability.prometheus import (
    ROLLING_SNAPSHOT_ROWS,
    TRAINING_GATING_REASONS_TOTAL,
    TRAINING_STEP_DURATION_SECONDS,
)
from app.storage.s3 import S3Storage
from app.training.fetch import FetchConfig
from app.training.gating import evaluate_publish_gate
from app.training.modeling import train_and_evaluate
from app.training.publish import (
    try_load_previous_metrics,
    update_latest_json,
    upload_model_artifacts_from_bytes,
)
from app.training.rolling import build_rolling_snapshot, ensure_month_snapshot, month_ranges
from app.training.snapshots import RawSnapshotPaths, load_trainable_rows_from_parquet
from app.training.versioning import make_model_version


@celery_app.task(
    name="app.tasks.rolling.fetch_month_snapshot",
    queue="training",
    base=TrainingTask,
)
def fetch_month_snapshot(start_date: str, end_date: str, force_fetch: bool = False) -> dict:
    started = time.perf_counter()
    try:
        storage = S3Storage()
        api_client = ApiClient()
        paths = ensure_month_snapshot(
            storage=storage,
            api_client=api_client,
            start=date.fromisoformat(start_date),
            end=date.fromisoformat(end_date),
            cfg=FetchConfig(),
            force_fetch=bool(force_fetch),
        )
        return {
            "start_date": paths.start_date,
            "end_date": paths.end_date,
            "raw_rows_key": paths.raw_rows_key,
            "manifest_key": paths.manifest_key,
            "prefix": paths.prefix,
        }
    finally:
        TRAINING_STEP_DURATION_SECONDS.labels(step="fetch_month_snapshot").observe(
            time.perf_counter() - started
        )


@celery_app.task(
    name="app.tasks.rolling.merge_rolling_12m",
    queue="training",
    base=TrainingShortTask,
)
def merge_rolling_12m(
    month_results: list[dict],
    as_of: str,
    months: int = 12,
    started_at: float | None = None,
) -> dict:
    started = time.perf_counter()
    try:
        storage = S3Storage()
        month_snapshots = [
            RawSnapshotPaths(
                prefix=res["prefix"],
                raw_rows_key=res["raw_rows_key"],
                manifest_key=res["manifest_key"],
                start_date=res["start_date"],
                end_date=res["end_date"],
            )
            for res in month_results
        ]

        paths, manifest = build_rolling_snapshot(
            storage=storage,
            month_snapshots=month_snapshots,
            as_of=date.fromisoformat(as_of),
            months=int(months),
        )

        counts = manifest.get("counts") or {}
        ROLLING_SNAPSHOT_ROWS.labels(metric="rows_raw_total").set(
            int(counts.get("rows_raw_total", 0))
        )
        ROLLING_SNAPSHOT_ROWS.labels(metric="rows_raw_deduped").set(
            int(counts.get("rows_raw_deduped", 0))
        )
        ROLLING_SNAPSHOT_ROWS.labels(metric="rows_trainable").set(
            int(counts.get("rows_trainable", 0))
        )

        result = {
            "snapshot_prefix": paths.prefix,
            "dataset_key": paths.dataset_key,
            "manifest_key": paths.manifest_key,
            "manifest": manifest,
        }
        if started_at is not None:
            result["started_at"] = float(started_at)

        return result
    finally:
        TRAINING_STEP_DURATION_SECONDS.labels(step="merge_rolling_12m").observe(
            time.perf_counter() - started
        )


@celery_app.task(
    name="app.tasks.rolling.train_rolling_12m",
    queue="training",
    base=TrainingLongTask,
    retry_kwargs={"max_retries": 2},
)
def train_rolling_12m(ctx: dict, train: bool = True) -> dict:
    started = time.perf_counter()
    try:
        if not train:
            ctx["train"] = {"skipped": True}
            return ctx

        storage = S3Storage()
        trainable_rows = load_trainable_rows_from_parquet(storage, ctx["dataset_key"])
        train_result = train_and_evaluate(trainable_rows)

        bio = BytesIO()
        joblib.dump(train_result.pipeline, bio)
        tmp_model_key = f"{ctx['snapshot_prefix']}/model_tmp.pkl"
        storage.put_bytes(
            bucket=settings.s3_bucket_snapshots,
            key=tmp_model_key,
            data=bio.getvalue(),
            content_type="application/octet-stream",
        )

        ctx["train"] = {
            "metrics": train_result.metrics,
            "feature_schema": train_result.feature_schema,
            "tmp_model_key": tmp_model_key,
        }
        return ctx
    finally:
        TRAINING_STEP_DURATION_SECONDS.labels(step="train_rolling_12m").observe(
            time.perf_counter() - started
        )


@celery_app.task(
    name="app.tasks.rolling.gate_rolling_12m",
    queue="training",
    base=TrainingShortTask,
)
def gate_rolling_12m(ctx: dict) -> dict:
    started = time.perf_counter()
    try:
        storage = S3Storage()
        manifest = ctx.get("manifest") or {}
        counts = manifest.get("counts") or {}
        rows_trainable = int(counts.get("rows_trainable", 0))
        metrics = ctx.get("train", {}).get("metrics") or {}
        prev_metrics = try_load_previous_metrics(storage)
        gate = evaluate_publish_gate(
            rows_trainable=rows_trainable,
            new_metrics=metrics,
            prev_metrics=prev_metrics,
        )
        ctx["gating"] = {
            "passed": gate.passed,
            "reasons": gate.reasons,
            "details": gate.details,
        }

        if gate.passed:
            TRAINING_GATING_REASONS_TOTAL.labels(reason="passed", status="passed").inc()
        elif gate.reasons:
            for reason in gate.reasons:
                TRAINING_GATING_REASONS_TOTAL.labels(reason=reason, status="failed").inc()
        else:
            TRAINING_GATING_REASONS_TOTAL.labels(reason="unknown", status="failed").inc()

        return ctx
    finally:
        TRAINING_STEP_DURATION_SECONDS.labels(step="gate_rolling_12m").observe(
            time.perf_counter() - started
        )


@celery_app.task(
    name="app.tasks.rolling.publish_rolling_12m",
    queue="training",
    base=TrainingTask,
)
def publish_rolling_12m(ctx: dict, publish: bool = True) -> dict:
    started = time.perf_counter()
    try:
        if not publish:
            ctx["published"] = {"skipped": True}
            return ctx

        gating = ctx.get("gating") or {}
        if not gating.get("passed", False):
            ctx["published"] = {
                "skipped": True,
                "reason": "gating_failed",
                "gating_reasons": gating.get("reasons", []),
            }
            return ctx

        train = ctx.get("train") or {}
        tmp_model_key = train.get("tmp_model_key")
        if not tmp_model_key:
            raise RuntimeError("Missing tmp_model_key for publish step")

        storage = S3Storage()
        pipeline_bytes = storage.get_bytes(bucket=settings.s3_bucket_snapshots, key=tmp_model_key)

        model_version = make_model_version()
        manifest = ctx.get("manifest") or {}
        training_manifest = {
            "model_version": model_version,
            "snapshot_prefix": ctx["snapshot_prefix"],
            "period": manifest.get("period"),
            "counts": manifest.get("counts"),
            "dropped_reasons": manifest.get("dropped_reasons"),
            "metrics": train.get("metrics"),
            "gating": gating,
        }

        try:
            keys = upload_model_artifacts_from_bytes(
                storage=storage,
                model_version=model_version,
                pipeline_bytes=pipeline_bytes,
                metrics=train.get("metrics") or {},
                feature_schema=train.get("feature_schema") or {},
                training_manifest=training_manifest,
            )
        finally:
            try:
                storage.delete(bucket=settings.s3_bucket_snapshots, key=tmp_model_key)
            except Exception as e:
                log().warning("temp_model_delete_failed", key=tmp_model_key, error=str(e))

        update_latest_json(
            storage=storage,
            model_version=model_version,
            artifact_key=keys["model_key"],
            snapshot_prefix=ctx["snapshot_prefix"],
        )

        ctx["published"] = {
            "attempted": True,
            "published": True,
            "latest_updated": True,
            "model_version": model_version,
            "model_key": keys["model_key"],
            "gating_passed": gating.get("passed", False),
            "gating_reasons": gating.get("reasons", []),
        }
        return ctx
    finally:
        TRAINING_STEP_DURATION_SECONDS.labels(step="publish_rolling_12m").observe(
            time.perf_counter() - started
        )
        started_at = ctx.get("started_at")
        if isinstance(started_at, (int, float)):
            TRAINING_STEP_DURATION_SECONDS.labels(step="rolling_chain_total").observe(
                max(0.0, time.time() - float(started_at))
            )
        started_at = ctx.get("started_at")
        if isinstance(started_at, (int, float)):
            TRAINING_STEP_DURATION_SECONDS.labels(step="rolling_chain_total").observe(
                max(0.0, time.time() - float(started_at))
            )


@celery_app.task(name="app.tasks.rolling.trigger_rolling_12m", queue="training", base=TrainingTask)
def trigger_rolling_12m(
    as_of: str | None = None,
    force_fetch: bool = False,
    train: bool = True,
    publish: bool = True,
    months: int = 12,
) -> dict:
    as_of_date = date.fromisoformat(as_of) if as_of else date.today()
    ranges = month_ranges(as_of_date, months=months)

    header = [
        fetch_month_snapshot.s(
            r.start.isoformat(), r.end.isoformat(), force_fetch=bool(force_fetch)
        )
        for r in ranges
    ]
    callback = (
        merge_rolling_12m.s(
            as_of=as_of_date.isoformat(),
            months=months,
            started_at=time.time(),
        )
        | train_rolling_12m.s(train=bool(train))
        | gate_rolling_12m.s()
        | publish_rolling_12m.s(publish=bool(publish and train))
    )

    result = chord(header)(callback)
    return {"task_id": result.id, "months": [r.start.isoformat() for r in ranges]}
