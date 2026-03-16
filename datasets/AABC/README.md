# AABC

Processed eval datasets are saved in [`data/processed`](data/processed/).

### `task5` eval dataset

`aabc-task5` is a balanced 5-way task-state evaluation dataset built from AABC `V1`
task-fMRI only. The class labels are:

- `go_hit`
- `nogo_correct`
- `encoding`
- `recall`
- `vismotor`

Build it locally with:

```bash
uv run python datasets/AABC/scripts/make_aabc_task5_arrow.py --space flat
```

This writes:

```text
datasets/AABC/data/processed/aabc-task5.flat.arrow
```

### Using `aabc_task5` in evals

The dataset registry name is:

```text
aabc_task5
```

By default, `fmri_fm_eval.datasets.aabc` resolves `aabc_task5` from the local repo
path first:

```text
datasets/AABC/data/processed/aabc-task5.{space}.arrow
```

If you set `AABC_ROOT`, it may point to either:

- the repo dataset root, e.g. `datasets/AABC`
- a processed-data root, e.g. `datasets/AABC/data/processed`
- a remote eval bucket root, e.g. `s3://medarc/fmri-datasets/eval`

Use `--dataset aabc_task5` with `main_probe.py` or `main_logistic.py`.
