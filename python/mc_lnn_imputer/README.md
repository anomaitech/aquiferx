# MC-LNN Imputer

Standalone GSLB imputation app that runs the validated two-stage pipeline:

1. `small-gap` imputation with `LNN-CFC auxiliary`
2. `large-gap` imputation with `iterative SoftImpute MC -> LNN refinement`

The package includes the required GSLB CSV inputs so it can run independently after install.

## Included data

- `datas/measurements_till_2023_to_lnn_imputation.csv`
- `datas/lnn_imputation_gslb_gldas_df_excercise.csv`

## Folder layout

- `app/`
  - `impute_gslb_full_iterative_softimpute.py`: standalone entrypoint
  - `run_gslb_long_gap_cv_mc_lnn.py`: base helper module
  - `run_gslb_long_gap_cv_mc_lnn_iterative_softimpute.py`: iterative MC+LNN helper module
  - `backend/lnn/`: required local backend package
- `datas/`: bundled CSV inputs
- `output/`: generated results

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

Impute all eligible GSLB wells:

```bash
python3 app/impute_gslb_full_iterative_softimpute.py
```

Impute selected wells only:

```bash
python3 app/impute_gslb_full_iterative_softimpute.py \
  --well-ids 415703112514501,414236112101201
```

Tune iterative refinement settings:

```bash
python3 app/impute_gslb_full_iterative_softimpute.py \
  --outer-iterations 2 \
  --feedback-prev-weight 0.35 \
  --support-frac 0.12 \
  --min-support 6 \
  --max-support 24
```

## Outputs

Files are written to `output/`:

- `full_imputed_series_<tag>.csv`
- `full_imputed_summary_<tag>.csv`
- `run_metadata_<tag>.json`
- `SUMMARY_<tag>.txt`

The series file contains:

- `raw_wte`
- `small_gap_wte`
- `final_wte`
- `fill_stage`

## Pipeline behavior

- Observed values are preserved.
- Small gaps are filled first using `LNN-CFC auxiliary`.
- Remaining large gaps are filled using iterative `SoftImpute MC -> LNN`.
- The iterative large-gap stage uses internal support-point selection from observed data to accept or reject refinement feedback.
