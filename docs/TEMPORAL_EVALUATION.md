# Temporal evaluation contract (version 1)

Training uses explicit calendar intervals, never a random fallback. Configure all
four settings (ISO dates) in the sync process or rolling trigger worker environment:

| Setting | Meaning |
| --- | --- |
| `REE_SPLIT_TRAIN_START` | Inclusive training start |
| `REE_SPLIT_VALIDATION_START` | Exclusive train end / inclusive validation start |
| `REE_SPLIT_TEST_START` | Exclusive validation end / inclusive test start |
| `REE_SPLIT_TEST_END` | Exclusive test end |

No production dates are supplied by this project. Choose dates appropriate to the snapshot and intended evaluation. All four boundaries must be
strictly increasing. Every normalized snapshot row must fall inside the configured
window: out-of-window rows cause an error rather than silent exclusion. Select a
compatible snapshot/window before training. Missing settings fail before sync
storage initialization or dispatch of rolling month-fetch tasks. Snapshot-only operations need no split
settings. The rolling trigger freezes boundary values in the training signature;
direct training-task calls resolve settings if no explicit config is supplied.
The CLI entry points use the same explicit settings.

## Split policy and preflight

Rows are normalized, then sorted by turnover date, property ID and canonical row
hash. Calendar dates are indivisible. There is no random stratification, segment
balancing, resampling or automatic cutoff selection.

Property IDs may repeat inside a partition (for example repeat sales); any ID
shared between train, validation and test rejects the entire split. Rows are not
moved between periods or dropped to hide overlap. This is deliberately stricter
than checking only exact sale rows, and may reject otherwise usable windows.

Before model construction/fit, reject invalid rows, invalid boundaries, empty
partitions, cross-partition property IDs, fewer than two distinct train prices,
and train features with no variation. The last two checks follow the current
CatBoost RMSE configuration. One validation or test row is mathematically usable;
it does not establish statistically meaningful quality. Counts by property type
are recorded independently in each partition. A missing segment means zero rows,
not an error or a reason to introduce random stratification. No new business
sample-size or quality thresholds have been selected.

`fit_model` fits a fresh CatBoost model on train, passing validation as the only
`eval_set` with early stopping. Test features and labels never enter that fit.
`evaluate_model` evaluates an already fitted model with its saved feature schema.
`train_and_evaluate` coordinates those operations and requires `split_config` in
addition to the explicit feature reference year. Its result now includes `lineage`. Direct gate callers must provide a verified
`comparison`; legacy `new_metrics`/`prev_metrics` arguments alone never authorize
promotion. Existing historical-metrics loading remains available for compatibility,
but neither orchestration path uses it for gating.

## Stored evidence

Both Pipeline and Celery carry the same `lineage` contract into the training
manifest. It records:

- Contract version, `fresh_fit` mode, boundaries and property overlap policy.
- Semantic normalized dataset fingerprint and snapshot manifest key/content,
  including the source revisions/lineage supplied by versioned snapshots.
- Per-partition count, observed period, sparse segment counts, dataset fingerprint,
  full membership (property ID, date, canonical full-row hash), and membership hash.
- Cross-partition ID intersections (empty for successful splits).

Membership includes labels and feature values through full-row hashes; equal
property/date pairs with different contents remain distinguishable. Duplicates
remain significant. Input permutation preserves membership and fingerprints;
training and evaluation use the canonical order. Membership can be checked against
the retained normalized snapshot without guessing which rows a model saw.
The result metrics include `n_train`, `n_validation`, `n_test`, test membership and
baseline results. Baseline parameters are global and per-type price medians fitted
only on train; types absent from train use the global median. Neither validation
nor test labels tune baseline parameters.

The sync path returns lineage even with publishing disabled; when publication is
requested, candidate artifacts include lineage/comparison even if the gate blocks
activation. Celery carries lineage/comparison in `ctx['train']`, serializes them
when publication is permitted, and retains them in the returned context when
blocked. This does not redesign artifact publication/retry/cleanup semantics.

## Champion/challenger comparison

Historical `metrics.json` is no longer the source for promotion comparisons.
The common comparison helper reads one champion pointer, its training manifest,
feature schema and model. Before reading model bytes or deserializing, the pointer
must have type `sklearn` and a string `model_version` matching
`[A-Za-z0-9][A-Za-z0-9._-]*` (a nonempty single path component). The artifact key
must be exactly `models/{model_version}/model.pkl`, and the manifest read from
`models/{model_version}/training_manifest.json` must contain the identical string
`model_version`. Schema is read from the same version prefix. Missing, invalid or
cross-version identity is unverifiable: no deserialization, prediction or automatic
promotion occurs, and missing identity is never inferred from other fields.
This binding does not establish content integrity, prevent overwriting objects
within a version, or protect against a concurrent champion change.

Each model transforms the exact same evaluation rows,
labels and order using its own saved schema/reference year. No intersection or
model-specific filtering is permitted; incorrect prediction counts/nonfinite
predictions reject evaluation. The supported schema remains feature contract v1;
unsupported schemas cause unverifiable comparison, never silent reinterpretation.

The candidate holdout must match the challenger's saved test membership and
fingerprint. Champion lineage must describe a supported fresh fit, all three
complete partitions with consistent membership counts/hashes/periods/boundaries,
and no forbidden internal property intersections. Every evaluation property must
be absent from champion train **and validation**. The earliest evaluation date
must also be strictly later than the last champion fit/validation exposure date.
No overlapping evaluation subset is silently removed. Champion's own test rows
are not fit/early-stopping exposure; reuse for promotion is selection exposure,
with the limitations below.

Successful comparison records common evaluation membership and actual row order,
fresh paired metrics, both feature schemas, champion pointer and champion lineage
fingerprint. These checks assume trusted artifacts and truthful complete fresh-fit
metadata; a checksum does not prove publisher trust or detect undeclared pretraining.
Warm-start models are unsupported by this lineage version.

An explicit `NoSuchKey` response for `latest.json` is the only storage result
interpreted as an absent active pointer. Other storage errors, unsupported lineage,
and invalid artifacts block promotion. This distinction depends on the provider's
error contract; it has not been verified against external providers.

## Promotion policy

`REE_PROMOTION_POLICY` is a JSON object; it is absent by default and is validated
when the gate runs, so inference can start without it. Missing or invalid policy
blocks automatic promotion. Configure every field explicitly. This example uses
arbitrary synthetic values only, not recommended thresholds:

```json
{"version":1,"bootstrap":false,"min_trainable":8,"min_evaluation":3,
 "segments":{"enebolig":1},"metrics":{
   "mdape":{"max":1.0,"max_degradation":0.1},
   "ae_p90":{"max":10000000,"max_degradation":0.1},
   "wape":{"max":1.0,"max_degradation":0.1}}}
```

All three metrics are required overall and for every listed segment. Limits are
nonnegative finite numbers; `max_degradation` is a fraction between 0 and 1.
Counts are positive integers and at least one required segment is mandatory.
Metrics are finite nonnegative JSON numbers, not strings or booleans. The supplied
trainable count must equal the sum of the train, validation and test lineage
counts; `min_trainable` applies to that full dataset. Counts, lineage, candidate
evaluation identity and paired comparison evidence must agree.
A zero champion value passes relative comparison only when the candidate is also
zero. Decisions contain stable reason codes and a canonical policy fingerprint;
they do not include exception text. With `bootstrap: true`, a confirmed absent
pointer may pass only the absolute and count checks. This flag is an explicit
request and defaults to disabled through the absent policy.

Legacy `REE_GATE_OVERALL_MAE_MAX_DEGRADE_PCT`,
`REE_GATE_ENEBOLIG_MAE_MAX_DEGRADE_PCT`, and the corresponding WAPE settings
are ignored; they are not mapped to the new rules. `REE_TRAIN_MIN_ROWS` is not
a promotion policy. Configure the complete new policy in each training process.
Existing queued training jobs must receive consistent configuration. No thresholds
are approved by this repository. Publication concurrency and retries remain outside
this contract.

## Compatibility and limits

- Existing callers must supply temporal configuration; random training and implicit
  dates are removed. Existing queued tasks without frozen boundaries require
  explicit worker settings and can fail; drain/requeue through the new trigger
  under a controlled rollout. No queues are changed automatically.
- Legacy models may still serve when their feature schema is supported, but cannot
  be used for verified comparison without complete exposure lineage. Do not invent
  lineage from aggregate historic metrics; use separately authorized retraining.
  Old artifacts are neither migrated nor deleted automatically.
- The holdout used for promotion participates in model selection. Repeated selection
  on it is not an independent final test. Independent quality claims require a
  separate subsequent untouched evaluation period/protocol.
- Historical README numbers came from the old random split and test-driven early
  stopping; they do not validate this temporal protocol. No new real-data quality
  results are claimed.
- Versioned snapshots preserve observed values/revisions, not proof that external
  features were historically available at each turnover date. Point-in-time feature
  availability and dataset/model rights require separate evidence.
- Fresh-fit lineage and paired evaluation do not solve concurrent champion changes
  between evaluation and pointer update. Conditional publication is a later stage.
- Offline tests use synthetic varied rows, spy models and in-memory storage. They
  do not establish real CatBoost fit, transport/provider behavior or deployment.
