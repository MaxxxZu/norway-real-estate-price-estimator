# Dataset, feature and snapshot contracts (v1)

These contracts address reproducible input data and feature parity. They do not
establish production readiness, real-world model quality, or historical
point-in-time correctness of features returned by the external API.

## Normalized training rows

`app.training.dataset.TrainingRow` defines normalization version 1. The canonical
property identity is a positive integer `id`; `turnover_date` is a calendar date
serialized as `YYYY-MM-DD`. These fields are required, including for direct
`train_and_evaluate` calls. Ingestion uses the matched property ID, turnover price
and turnover date as authoritative values; estimation parameters cannot overwrite
them. Fractional turnover prices are preserved rather than truncated.

Numeric strings and integral numeric values are normalized through Pydantic:
IDs, municipality, construction year and counts become integers; price, location
and areas become floats. Booleans, fractional integers, NaN and infinities are
rejected. Price and areas must be positive; total area must be at least BRA.
Municipality must be positive, latitude/longitude must be within geographic
bounds, construction year must be 1800–2100, rooms/bedrooms must be nonnegative,
and type must be a supported `RealEstateType`. These retain the existing feature
contract's domain bounds, without introducing model-quality thresholds.

Floor is required for apartments. Other absent optional fields become explicit
`null`, including floor for other property types. Unknown fields are discarded;
`id` is authoritative rather than falling back to `remote_id`/`property_id`.
Output is a new normalized mapping; inputs are not mutated. One deterministic
first failure is counted per rejected row (`missing:<field>`, `invalid:<field>`,
with explicit cross-field reasons). These are contract failures, not quality
metrics. Rejected raw rows remain in snapshot source files.

Rolling selection validates identity/date first, then selects the latest calendar
date per property and validates the selected row's remaining fields. An invalid
latest transaction does not fall back to an older transaction. Same-day conflicts
select the lexicographically greatest canonical row, independent of source/row
order; this is a reproducibility tie-break, not evidence of transaction ordering.
Identity failures are counted as `invalid_identity:<field>`. `rows_raw_total`
counts all source rows, `rows_raw_deduped` counts selected identity-valid rows,
and `rows_trainable` counts fully normalized selected rows. Normalization changes,
including tie-break changes, require a normalization-version increment.

## Features and models

`app.features` is the shared transformer. Schema version 1 explicitly records
ordered categorical/numeric/derived columns, logical dtypes, missing-value
representation, transforms, and `feature_reference_year`. Categorical columns are
Python strings in object columns (including municipality); every numeric column
is float64. Missing optional numerics become NaN for CatBoost. Derived values are
always rebuilt by the shared transformer; upstream derived columns are ignored.
The transformer expects validated feature values (normalized training rows or
`EstimationFeatures` in serving); it does not replace API validation.

`area_ratio = bra / total_area` and
`building_age = feature_reference_year - built_year`. Neither depends on the
system clock. New training calls require an explicit keyword `reference_year`.
Both synchronous and Celery orchestration use the year of the snapshot period's
end date. The returned versioned `feature_schema.json` is published using the
existing artifact writer and loaded by the registry. Serving uses its stored
year for the lifetime of that model. Construction years after the reference year
can yield negative ages, preserving the existing domain bounds; this is not a
point-in-time feature reconstruction.

Schema loading rejects missing/unknown versions, missing reference years,
changed columns/order/dtypes/missing policies or unsupported transforms before
joblib deserialization. Version 1 accepts the exact schema emitted by the
builder. Adding schema metadata requires an explicit contract update. Joblib
artifacts still require a trusted publisher/storage boundary; schema validation
and checksums do not authenticate a publisher. Serialization format, temporal
split, early stopping, quality gate and registry refresh policy are unchanged.

## Immutable storage revisions and discovery

New objects use `snapshots/v1/`. Existing unversioned objects are untouched.

- Monthly source: `snapshots/v1/<start>_<end>/revisions/<uuid>/` contains
  `rows_raw.jsonl` and `manifest.json`.
- Monthly discovery: `snapshots/v1/<start>_<end>/latest_raw.json` contains a
  revision ID. Non-forced fetch reuses only a completed compatible revision.
  Force-fetch always allocates a new revision, even for identical content.
- Synchronous dataset: `snapshots/v1/<start>_<end>/revisions/<uuid>/` contains
  raw rows, normalized `dataset.parquet` and a completion manifest.
- Rolling dataset: `snapshots/v1/rolling/<window>/revisions/<uuid>/` contains
  the complete raw source multiset, normalized selected dataset and manifest.
  The manifest retains exact source revisions, object keys, raw fingerprints,
  source normalization versions and stable `dataset_identity`.
- Rolling discovery: `snapshots/v1/rolling/<window>/identities/<identity>.json`
  points to a completed revision. Identity hashes the window, number of months,
  normalization version and ordered exact source references. Changing any source
  revision invalidates reuse even when normalized content is unchanged.

Every writer uploads data files first and writes a completion manifest last
inside the revision. Discovery pointers are updated only after completion. A
manifest records `snapshot_version`, `normalization_version`, `revision`,
`complete: true` and raw/normalized semantic fingerprints as applicable. Readers
require compatible completion metadata and all expected data files. Source data
and loaded normalized Parquet rows are fingerprint-checked before use. Nullable
Parquet values are restored to `None` and normalized again on reading.

Each orchestration attempt owns a new UUID prefix, including concurrent attempts
for the same logical rolling identity. They may complete separate equivalent
revisions; the discovery pointer may select either. No writer mutates historical
revision files. A repeated explicit upload to a completed prefix checks content
and lineage and returns without writing, or raises an immutable conflict. Writers
must allocate their own prefixes; this is not storage-level object locking/WORM
or a conditional-write protocol for arbitrary callers sharing a prefix.

A crash before completion leaves unusable files; a retry creates a new revision.
A crash after completion but before discovery leaves a complete orphan; a retry
may create another complete revision. Neither partial files nor old revisions
are deleted automatically. Missing data behind a discovery entry is not reused;
rolling rebuilds into another revision. Storage/read failures propagate rather
than being interpreted as a cache miss. Storage conditional writes, transport
failures and provider-specific consistency need separate integration evidence.
Publisher concurrency and temporary-model cleanup remain separate later work.

Fingerprints are SHA-256 over canonical UTF-8 JSON: sorted object keys, sorted
row multiset, significant duplicate multiplicity and normalized equivalent
integer/float numeric representations. Nested list order remains significant.
Raw non-finite floats use a deterministic tagged representation for hashing
with actual objects wrapped separately to prevent reserved-tag ambiguity
while raw JSONL preserves the original rejected values (Python JSON extensions
NaN/Infinity). Normalized dataset JSON contains no non-finite values. Fingerprints
are integrity checks relative to metadata, not publisher trust. Reproducibility
means equivalent canonical data, not identical Parquet bytes or model bytes.

## Compatibility and transition

The versioned training/serving readers reject unsupported legacy snapshots and
model schemas. There is no automatic migration, guessed reference year,
calendar-dependent fallback, deletion or activation.

Legacy/unversioned snapshots cannot be reused by new readers. Fetching a new
versioned snapshot is a separately authorized ingestion operation. Legacy
sklearn/CatBoost artifacts whose feature schemas lack the version/year contract
raise `ModelNotReadyError` in the registry. Existing stub artifacts retain their
independent stub contract; they do not use this feature transformer.

Deploying this reader with an active legacy learned model can make estimation
unavailable. Prepare compatible data and a retrained model, review its quality and plan
activation before switching serving code; alternatively retain the previous
code/artifact pair during the transition. Reader upgrades do not perform these
operations automatically. Never fabricate an old model's reference year by
editing metadata; no legacy compatibility adapter is provided.

Snapshots preserve what was fetched, not what was historically observable on a
transaction date. External features may reflect later property state. Historical
feature availability needs separate evidence. The
[temporal evaluation protocol](TEMPORAL_EVALUATION.md) separates data by date;
neither it nor frozen snapshots reconstruct historical external feature values.
