# PRO-cap Training Benchmark

This folder contains self-contained training code for CAPY and ProCapNet on
processed PRO-cap data. It does not import from `examples/procap/ProCapNet` at
runtime.

Default data root:

```bash
/grid/koo/home/shared/capybara/procap
```

Run one CAPY fold:

```bash
bash examples/procap/run_train_capy.sh configs/default_procap.yaml "" K562 procap 1 0
```

Run one ProCapNet fold:

```bash
bash examples/procap/run_train_procapnet.sh "" K562 procap 1 0
```

Submit all folds for K562:

```bash
bash examples/procap/submit_train_all_folds_capy.sh
bash examples/procap/submit_train_all_folds_procapnet.sh
```

Set `PROCAP_PROJ_DIR` to override the processed data root. Set `FOLDS`, for
example `FOLDS="1"`, to limit the all-fold wrappers.

The final numeric argument in the shell wrappers is exported as
`CUDA_VISIBLE_DEVICES` before Python starts. The Python entrypoints also accept
`--device`, which defaults to `gpu`; this errors if CUDA is unavailable. Use
`--device cpu` or `--device auto` for lenient local checks.
