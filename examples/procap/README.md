# PRO-cap Benchmark

This folder contains self-contained training, evaluation, and benchmarking code
for CAPY and ProCapNet on processed PRO-cap data. The training and evaluation
entrypoints do not import from `examples/procap/ProCapNet` at runtime.

Default data root:

```text
/grid/koo/home/shared/capybara/procap
```

Set `PROCAP_PROJ_DIR` to override the processed data root. The shell scripts in
this directory are mainly internal wrappers for local runs and cluster
submissions; use them as references if you need concrete argument examples.

## Overview

The PRO-cap workflow has four main steps:

1. Generate or obtain processed PRO-cap inputs using the upstream ProCapNet data
   workflow.
2. Train CAPY or ProCapNet with the Python training entrypoints.
3. Evaluate trained checkpoints with the Python evaluation entrypoint.
4. Compare saved predictions and metrics in the benchmark notebook.

## Results

The benchmark tables below summarize K562 PRO-cap results from
`examples/procap/results/benchmark_250513`. Values are fold mean +/- sample std
across 7 folds. For JSD, lower is better; for Pearson, Spearman, and R2, higher
is better.

### Profile Metrics

| Metric | ProCapNet | CAPY | CAPY change |
| --- | ---: | ---: | ---: |
| JSD | 0.693 +/- 0.004 | 0.653 +/- 0.004 | -0.039 |
| Pearson | 0.540 +/- 0.006 | 0.595 +/- 0.006 | +0.054 |
| Spearman | 0.292 +/- 0.004 | 0.304 +/- 0.004 | +0.011 |

### Count Metrics

| Split | Metric | ProCapNet | CAPY | CAPY change |
| --- | --- | ---: | ---: | ---: |
| Positive peaks | Pearson | 0.719 +/- 0.025 | 0.743 +/- 0.022 | +0.023 |
| Positive peaks | Spearman | 0.727 +/- 0.021 | 0.769 +/- 0.018 | +0.042 |
| Positive peaks | R2 | 0.446 +/- 0.073 | 0.327 +/- 0.070 | -0.120 |
| Negative peaks | Pearson | 0.358 +/- 0.033 | 0.499 +/- 0.018 | +0.141 |
| Negative peaks | Spearman | 0.324 +/- 0.037 | 0.492 +/- 0.017 | +0.168 |
| Negative peaks | R2 | -2.892 +/- 0.312 | -0.789 +/- 0.170 | +2.102 |
| Positive + negative peaks | Pearson | 0.726 +/- 0.015 | 0.759 +/- 0.007 | +0.033 |
| Positive + negative peaks | Spearman | 0.623 +/- 0.030 | 0.728 +/- 0.017 | +0.105 |
| Positive + negative peaks | R2 | -0.087 +/- 0.073 | 0.467 +/- 0.044 | +0.554 |

CAPY improves profile metrics and the combined positive/negative count metrics.
Positive-peak count R2 is the main exception in this run, while positive-peak
count correlations still improve.

## 1. Generate PRO-cap Data

This example expects the processed PRO-cap directory layout used by ProCapNet:
genome files, annotations, per-cell-type processed BEDs, and plus/minus
BigWigs.

Use the upstream ProCapNet workflow as the source of truth for generating these
files:

```text
https://github.com/kundajelab/ProCapNet/
```

In that repository, start with the scripts under:

```text
src/0_download_files
src/1_process_data
```

Those steps create the inputs consumed here, including files such as:

```text
genome/hg38.withrDNA.fasta
genome/hg38.withrDNA.chrom.sizes
annotations/hg38.k36.multiread.umap.bigWig
data/procap/processed/<CELL_TYPE>/peaks_fold<FOLD>_train.bed.gz
data/procap/processed/<CELL_TYPE>/peaks_fold<FOLD>_val.bed.gz
data/procap/processed/<CELL_TYPE>/peaks_fold<FOLD>_test.bed.gz
data/procap/processed/<CELL_TYPE>/5prime.pos.bigWig
data/procap/processed/<CELL_TYPE>/5prime.neg.bigWig
```

## 2. Train Models

Use these Python entrypoints for model training:

- `examples/procap/train_capy.py` trains CAPY.
- `examples/procap/train_procapnet.py` trains the local ProCapNet baseline.

Both scripts accept the same core arguments:

- `--proj_dir`: processed PRO-cap project directory.
- `--params`: YAML parameter file.
- `--cell_type`: cell type, such as `K562`.
- `--data_type`: data namespace, usually `procap`.
- `--fold`: fold number.
- `--timestamp`: optional run identifier.
- `--stage`: `train`, `finetune`, or `both`.
- `--device`: `gpu`, `cpu`, `auto`, or a specific PyTorch device string.

CAPY defaults to `configs/default_procap.yaml`; ProCapNet defaults to
`configs/default_procapnet.yaml`.

The `--stage` argument controls how training is staged:

- `train`: train the model once from initialization and write outputs under the
  requested timestamp.
- `finetune`: load an existing stage-1 run and fine-tune the count-related
  parameters, writing outputs under `<timestamp>_ft`.
- `both`: run `train` first, then immediately run the count fine-tuning stage.

Fine-tuning is optional. In our benchmarks, it sometimes improved count R2 on
positive peaks a little, but we did not see a significant overall improvement.
Fine-tune-only runs require an existing stage-1 timestamp.

Training outputs are written under:

```text
models/<model_name>/<data_type>/<cell_type>/fold<fold>/<timestamp>/
```

Each run writes `params_saved.yaml`, `config_saved.yaml`, `metrics.tsv`, and
checkpoints under `checkpoints/`. Evaluation uses `checkpoints/best.pt`.
Two-stage or fine-tune-only runs write the fine-tuned model under:

```text
models/<model_name>/<data_type>/<cell_type>/fold<fold>/<timestamp>_ft/
```

The shell wrappers `run_train_capy.sh`, `run_train_procapnet.sh`,
`submit_train_all_folds_capy.sh`, and `submit_train_all_folds_procapnet.sh`
show how local batch runs have been launched, but they are not the primary
interface documented here.

## 3. Evaluate Trained Models

Use `examples/procap/evaluate.py` to evaluate trained CAPY or ProCapNet
checkpoints on a split.

Important arguments:

- `--proj_dir`: processed PRO-cap project directory.
- `--model_name`: `capy` or `procapnet`.
- `--cell_type`: cell type, such as `K562`.
- `--data_type`: data namespace, usually `procap`.
- `--fold`: fold number.
- `--timestamp`: trained run identifier, including `_ft` for fine-tuned runs.
- `--split`: one of `train`, `val`, `test`, `all`, `dnase_train`,
  `dnase_val`, or `dnase_test`.
- `--reverse_complement`: enable reverse-complement test-time augmentation.
- `--save_predictions`: save predicted log profiles and counts as `.npy`
  files.
- `--device`: `gpu`, `cpu`, `auto`, or a specific PyTorch device string.

Evaluation outputs are written directly under the run evals directory:

```text
<cell_type>_metrics_summary_<split>.csv
<cell_type>_metrics_profile_<split>.csv
<cell_type>_eval_log_<split>.txt
<cell_type>_log_pred_profiles_<split>.npy
<cell_type>_log_pred_counts_<split>.npy
```

Reverse-complement augmented runs add `_rc` before the split:

```text
<cell_type>_metrics_summary_rc_<split>.csv
<cell_type>_log_pred_profiles_rc_<split>.npy
```

The shell wrappers `run_evaluate.sh` and `submit_evaluate_all_folds.sh` are
available as references for existing local workflows.

## 4. Benchmark and Plot

After evaluating the folds you want to compare, open:

```text
examples/procap/benchmark_procap.ipynb
```

Edit the notebook config cell:

```python
proj_dir = Path("/grid/koo/home/shared/capybara/procap")
cell_type = "K562"
data_type = "procap"
split = "test"
folds = [1, 2, 3, 4, 5, 6, 7]

model_specs = [
    {
        "label": "ProCapNet",
        "model_name": "procapnet",
        "timestamp": "260409_procapnet",
        "color": "#6B7280",
    },
    {
        "label": "CAPY",
        "model_name": "capy",
        "timestamp": "260409_capy",
        "color": "#0F766E",
    },
]
```

The notebook loads saved `.npy` predictions from each run evals directory,
computes profile/count metrics, and generates ProCapNet-style benchmark figures.
