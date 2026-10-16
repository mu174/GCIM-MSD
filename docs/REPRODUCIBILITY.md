# Reproducibility record

## Main-run protocol

- one recorded run per dataset;
- deterministic seed setup, including `PYTHONHASHSEED` before process start;
- 30 epochs;
- gradient accumulation over two batches;
- linear learning-rate schedule with five warmup optimizer updates;
- checkpoint selection on the validation split only;
- selection score: `0.7 * sarcasm weighted F1 + 0.3 * sentiment weighted F1`;
- sarcasm weighted F1 breaks an exact joint-score tie;
- one test evaluation after reloading the selected checkpoint.

MUStARD used inverse-frequency sentiment loss weights computed from its
training split. Memotion used an unweighted sentiment cross-entropy loss.

## Training loss

```text
L = 0.5 * L_sarcasm + 0.5 * L_sentiment + 1.0 * L_MCM
```

The MCM margin is `0.5` in both configurations.

## Metric definition

For class `c`, let `n_c` be its support and `N` the number of test examples.
The reported weighted quantities are:

```text
weighted precision = sum_c (n_c / N) * precision_c
weighted recall    = sum_c (n_c / N) * recall_c
weighted F1        = sum_c (n_c / N) * F1_c
```

They are read from the `weighted avg` entry of scikit-learn's
`classification_report`. For these single-label tasks, weighted recall equals
accuracy.

## Scope of the release

The `src/` implementation is derived from the immutable source snapshot of
run `20261015_074731_489865_joint70_30_both_mu_balanced_me_none`. Changes in
this release include filenames, portable paths, removal of an unused
spaCy import, documentation, standalone evaluation, config-driven launchers,
explicit device selection, and run archiving. The model, loss,
optimizer, scheduling, metric, and checkpoint-selection operations are
retained.

## Release launcher

The shell launchers read configs/<dataset>.yaml through scripts/launch.py.
Architecture values are fixed in model source, not configurable YAML fields.
Unknown configuration keys or unsupported selection policies cause an error.
CPU is the default, matching the archived run. To select CUDA or an alternative
configuration, call scripts/launch.py with --device cuda or --config PATH.
Use --plan-only to inspect resolved settings without creating a run.

Each new run directory includes source_snapshot/, config.yaml, RUN_SPEC.json,
STATUS.json and per-job train.log, artifacts/best_model.pth and metrics JSON.
Logs stream unbuffered. Existing run directories are never reused.
These release wrappers were checked without retraining; identical numerical
results on other hardware or dependency builds are not guaranteed.

The released source is a portable adaptation. Model and engine files remain
unchanged; path handling, launchers, device selection and metadata plumbing
are release changes, not additional archived experiments.
