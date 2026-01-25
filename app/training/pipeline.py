from datetime import date
from typing import Any

from app.clients.api_client import ApiClient
from app.storage.s3 import S3Storage
from app.training.dataset import DatasetBuildResult, build_trainable_dataset
from app.training.fetch import (
    FetchConfig,
    build_properties,
    build_rows,
    fetch_estimation_params,
    fetch_turnovers,
    normalize_turnovers,
)
from app.training.gating import evaluate_publish_gate
from app.training.modeling import train_and_evaluate
from app.training.publish import (
    try_load_previous_metrics,
    update_latest_json,
    upload_model_artifacts,
)
from app.training.snapshots import (
    SnapshotPaths,
    fetch_latest_snapshot_ref,
    load_manifest,
    load_trainable_rows_from_parquet,
    upload_snapshots,
)
from app.training.versioning import make_model_version


class Pipeline:
    def __init__(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        dry_run: bool = False,
        train: bool = True,
        publish: bool = True,
        force_fetch: bool = False
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.dry_run = dry_run
        self.train = train
        self.publish = publish
        self.force_fetch = force_fetch
        self._validate_params()
        self.storage = S3Storage()
        self.api_client = ApiClient()
        self.cfg = FetchConfig()
        self.result: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}
        self.paths: SnapshotPaths = None

    def process(self) -> dict:
        self._build_dataset()
        self._fetch_latest_snapshot()
        self._build_initial_result()
        train_result = self._train_model()
        self._publish_model(train_result)

        return self.result

    def _validate_params(self) -> None:
        if self.dry_run and self.train:
            raise ValueError("dry_run and train are mutually exclusive")

        if self.publish and not self.train:
            raise ValueError("publish=True requires train=True")

        if not self.force_fetch:
            return

        if self.start_date is None or self.end_date is None:
            raise ValueError("force_fetch=True requires start_date and end_date")

        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")

        return

    def _build_dataset(self) -> None:
        if not self.force_fetch:
            return

        turnovers_raw = fetch_turnovers(self.api_client, self.start_date, self.end_date, self.cfg)
        turnovers = normalize_turnovers(turnovers_raw)

        cadastral_unit_ids = [t["cadastral_unit_ids"][0] for t in turnovers]
        properties = build_properties(self.api_client, cadastral_unit_ids)
        estimation_params = fetch_estimation_params(self.api_client, properties)
        rows_raw = build_rows(turnovers, properties, estimation_params)

        dataset_result = build_trainable_dataset(rows_raw)
        self._add_snapshot(
            dataset_result, turnovers_raw, turnovers, cadastral_unit_ids, properties, rows_raw
        )

    def _add_snapshot(
        self,
        dataset_result: DatasetBuildResult,
        turnovers_raw: list[dict],
        turnovers: list[dict],
        cadastral_unit_ids: list[str],
        properties: list[dict],
        rows_raw: list[dict],
    ) -> None:
        self.manifest = {
            "period": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat()
            },
            "counts": {
                "turnovers_raw": len(turnovers_raw),
                "turnovers_normalized": len(turnovers),
                "cadastral_unit_ids": len(cadastral_unit_ids),
                "properties_matched": len(properties),
                "rows_raw": len(rows_raw),
                "rows_trainable": len(dataset_result.trainable_rows),
            },
            "dropped_reasons": dataset_result.dropped_reasons,
            "dry_run": bool(self.dry_run),
            "train": bool(self.train),
            "publish": bool(self.publish),
        }

        self.paths = upload_snapshots(
            storage=self.storage,
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
            raw_rows=rows_raw,
            trainable_rows=dataset_result.trainable_rows,
            manifest=self.manifest,
        )

    def _fetch_latest_snapshot(self) -> None:
        if self.force_fetch:
            return

        self.paths = fetch_latest_snapshot_ref(self.storage)
        self.manifest = load_manifest(self.storage, self.paths.manifest_key)

    def _build_initial_result(self) -> None:
        self.result = {
            "snapshots": {
                "raw_rows_key": self.paths.raw_rows_key,
                "dataset_key": self.paths.dataset_key,
                "manifest_key": self.paths.manifest_key,
            },
            "manifest": self.manifest,
        }

    def _train_model(self) -> Any:
        if not self.train:
            return

        trainable_rows = load_trainable_rows_from_parquet(
            self.storage,
            self.paths.dataset_key,
        )
        train_result = train_and_evaluate(trainable_rows)

        self.result["metrics"] = train_result.metrics

        return train_result

    def _publish_model(self, train_result) -> None:
        if not self.publish:
            return

        model_version = make_model_version()
        snapshot_prefix = self.paths.prefix
        prev_metrics = try_load_previous_metrics(self.storage)
        gate = evaluate_publish_gate(
            rows_trainable=int(self.manifest["counts"]["rows_trainable"]),
            new_metrics=train_result.metrics,
            prev_metrics=prev_metrics,
        )

        training_manifest = {
            "model_version": model_version,
            "snapshot_prefix": snapshot_prefix,
            "period": self.manifest["period"],
            "counts": self.manifest["counts"],
            "dropped_reasons": self.manifest["dropped_reasons"],
            "metrics": train_result.metrics,
            "gating": {
                "passed": gate.passed,
                "reasons": gate.reasons,
                "details": gate.details,
            },
        }

        keys = upload_model_artifacts(
            storage=self.storage,
            model_version=model_version,
            pipeline=train_result.pipeline,
            metrics=train_result.metrics,
            feature_schema=train_result.feature_schema,
            training_manifest=training_manifest,
        )
        published = False
        latest_updated = False
        if gate.passed:
            update_latest_json(
                storage=self.storage,
                model_version=model_version,
                artifact_key=keys["model_key"],
                snapshot_prefix=snapshot_prefix,
            )
            published = True
            latest_updated = True

        self.result["published"] = {
            "attempted": True,
            "published": published,
            "latest_updated": latest_updated,
            "model_version": model_version,
            "model_key": keys["model_key"],
            "gating_passed": gate.passed,
            "gating_reasons": gate.reasons,
        }
