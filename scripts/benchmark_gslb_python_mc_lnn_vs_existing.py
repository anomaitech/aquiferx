#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/tmp/aquiferx_repo")
DETAIL_CSV = ROOT / "benchmark_outputs" / "gslb_longgap_browser_imputer_benchmark_by_well_repeat.csv"
OUT_DIR = ROOT / "benchmark_outputs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def compute_metrics(obs: List[float], pred: List[float]) -> Dict[str, float]:
    arr_o = np.asarray(obs, dtype=float)
    arr_p = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(arr_p - arr_o)))
    mse = float(np.mean((arr_p - arr_o) ** 2))
    rmse = float(np.sqrt(mse))
    obs_mean = float(np.mean(arr_o))
    ss_tot = float(np.sum((arr_o - obs_mean) ** 2))
    ss_res = float(np.sum((arr_p - arr_o) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "n": float(len(obs))}


def main() -> None:
    app_dir = ROOT / "python" / "mc_lnn_imputer" / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    base = load_module(
        "mc_lnn_base",
        app_dir / "run_gslb_long_gap_cv_mc_lnn.py",
    )
    soft = load_module(
        "mc_lnn_soft",
        app_dir / "run_gslb_long_gap_cv_mc_lnn_iterative_softimpute.py",
    )

    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"Missing benchmark fold file: {DETAIL_CSV}")

    detail = pd.read_csv(DETAIL_CSV)
    required_cols = {
        "duration_years", "repeat", "well_id", "start_idx", "start_date", "end_date",
        "withheld_n", "original_r2", "original_rmse"
    }
    missing = required_cols - set(detail.columns)
    if missing:
        raise ValueError(f"Benchmark fold file missing columns: {sorted(missing)}")

    main_df, aux_df, months, _top_wells, donor_pool_wells, latlon = base.load_data()
    aux_lk = base.aux_lookup(aux_df)
    params = base.build_params()
    raw_obs = {wid: base.monthly_series(main_df, wid, months) for wid in donor_pool_wells}
    donor_prefilled_all = base.prefill_small_gaps_all(
        raw_obs,
        months,
        aux_lk,
        latlon,
        params,
        base._rng_seed(42, "prefill_all_python_benchmark"),
    )

    out_rows: List[Dict[str, Any]] = []
    summary_lines: List[str] = [
        "GSLB Long-Gap Benchmark: Existing AquiferX vs Validated Python SoftImpute MC+LNN",
        "Date overlap: 2000-01-01 to 2023-12-01 (2024 unavailable in bundled observed/GLDAS data)",
        "",
    ]

    for years in sorted(detail["duration_years"].unique()):
        sub = detail[detail["duration_years"] == years].copy()
        existing_obs_all: List[float] = []
        existing_pred_all: List[float] = []
        python_obs_all: List[float] = []
        python_pred_all: List[float] = []

        for row in sub.itertuples(index=False):
            wid = str(row.well_id)
            if wid not in raw_obs:
                continue

            raw_target = list(raw_obs[wid])
            start_idx = int(row.start_idx)
            gap_months = int(years) * 12
            window_indices = list(range(start_idx, min(len(raw_target), start_idx + gap_months)))
            truth = {i: float(raw_target[i]) for i in window_indices if raw_target[i] is not None}
            if not truth:
                continue

            masked_raw_target = list(raw_target)
            for i in window_indices:
                masked_raw_target[i] = None

            target_prefilled = base.prefill_small_gaps_all(
                {wid: masked_raw_target},
                months,
                aux_lk,
                latlon,
                params,
                base._rng_seed(42, "prefill_target_python_benchmark", wid, int(years), int(row.repeat), start_idx),
            )[wid]

            donor_obs_fold = {k: list(v) for k, v in donor_prefilled_all.items()}
            donor_obs_fold[wid] = list(target_prefilled)

            result, meta = soft.run_iterative_fold(
                base,
                wid,
                raw_target,
                list(target_prefilled),
                donor_obs_fold,
                months,
                aux_lk,
                latlon,
                params,
                seed=base._rng_seed(42, "python_softimpute_benchmark", wid, int(years), int(row.repeat), start_idx),
                outer_iterations=2,
                feedback_prev_weight=0.35,
                support_frac=0.12,
                min_support=6,
                max_support=24,
            )
            py_metrics = base.metrics_on_truth(result, truth)
            if not py_metrics:
                continue

            obs_vals: List[float] = []
            existing_pred_vals: List[float] = []
            python_pred_vals: List[float] = []
            for idx, tv in truth.items():
                pred_py = result[idx].imputed if idx < len(result) and result[idx].imputed is not None else None
                if pred_py is None or not np.isfinite(float(pred_py)):
                    continue
                obs_vals.append(float(tv))
                python_pred_vals.append(float(pred_py))

            if not obs_vals:
                continue

            existing_r2 = float(row.original_r2)
            existing_rmse = float(row.original_rmse)
            existing_ss_tot = float(np.sum((np.asarray(obs_vals) - np.mean(obs_vals)) ** 2))
            existing_sse = max(0.0, (existing_rmse ** 2) * len(obs_vals))
            # Reconstruct a compatible aggregate only for summary accounting.
            # Detailed row keeps the original benchmark metrics verbatim.
            existing_obs_all.extend(obs_vals)
            # Use a constant-offset reconstruction is not valid pointwise; keep pooled existing metrics separately.
            # For fair summary by duration, average row metrics below.
            python_obs_all.extend(obs_vals)
            python_pred_all.extend(python_pred_vals)

            out_rows.append({
                "duration_years": int(years),
                "repeat": int(row.repeat),
                "well_id": wid,
                "start_idx": start_idx,
                "start_date": row.start_date,
                "end_date": row.end_date,
                "withheld_n": int(row.withheld_n),
                "existing_r2": existing_r2,
                "existing_rmse": existing_rmse,
                "python_r2": float(py_metrics.get("r2", np.nan)),
                "python_rmse": float(py_metrics.get("rmse", np.nan)),
                "python_mae": float(py_metrics.get("mae", np.nan)),
                "python_mse": float(py_metrics.get("mse", np.nan)),
                "outer_iterations_used": int(meta.get("outer_iterations_used", 0)),
                "support_points": int(meta.get("support_points", 0)),
                "best_support_kge": float(meta.get("best_support_kge", np.nan)),
                "best_support_rmse": float(meta.get("best_support_rmse", np.nan)),
            })

        duration_rows = [r for r in out_rows if r["duration_years"] == int(years)]
        if not duration_rows:
            continue
        existing_r2_mean = float(np.mean([r["existing_r2"] for r in duration_rows]))
        existing_rmse_mean = float(np.mean([r["existing_rmse"] for r in duration_rows]))
        python_r2_mean = float(np.mean([r["python_r2"] for r in duration_rows]))
        python_rmse_mean = float(np.mean([r["python_rmse"] for r in duration_rows]))
        python_mae_mean = float(np.mean([r["python_mae"] for r in duration_rows]))
        python_mse_mean = float(np.mean([r["python_mse"] for r in duration_rows]))

        summary_lines.append(f"{int(years)}y long gap")
        summary_lines.append(
            f"Existing PCHIP + ELM: rows={len(duration_rows)}, mean_R2={existing_r2_mean:.4f}, mean_RMSE={existing_rmse_mean:.4f}"
        )
        summary_lines.append(
            f"Validated Python SoftImpute MC+LNN: rows={len(duration_rows)}, mean_R2={python_r2_mean:.4f}, mean_RMSE={python_rmse_mean:.4f}, mean_MAE={python_mae_mean:.4f}, mean_MSE={python_mse_mean:.4f}"
        )
        summary_lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_out = OUT_DIR / "gslb_longgap_python_softimpute_vs_existing_by_well_repeat.csv"
    summary_out = OUT_DIR / "gslb_longgap_python_softimpute_vs_existing_summary.txt"
    pd.DataFrame(out_rows).to_csv(detail_out, index=False)
    summary_out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"Wrote:\n- {summary_out}\n- {detail_out}")


if __name__ == "__main__":
    main()
