#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
OUT_DIR = ROOT_DIR / "output"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def impute_one_well(
    base,
    iterative,
    wid: str,
    raw_target: List[Optional[float]],
    target_series: List[Optional[float]],
    donor_obs_filled: Dict[str, List[Optional[float]]],
    months: pd.DatetimeIndex,
    aux_lk: Dict[Tuple[int, int], List[float]],
    latlon: Dict[str, Tuple[float, float]],
    params,
    seed: int,
    outer_iterations: int,
    feedback_prev_weight: float,
    support_frac: float,
    min_support: int,
    max_support: int,
) -> Tuple[List[Optional[float]], List[str], Dict[str, Any]]:
    mod_target = list(target_series)
    donor_obs_fold = {k: list(v) for k, v in donor_obs_filled.items()}
    donor_obs_fold[wid] = mod_target

    result, iter_meta = iterative.run_iterative_fold(
        base,
        wid,
        raw_target,
        mod_target,
        donor_obs_fold,
        months,
        aux_lk,
        latlon,
        params,
        seed=seed,
        outer_iterations=outer_iterations,
        feedback_prev_weight=feedback_prev_weight,
        support_frac=support_frac,
        min_support=min_support,
        max_support=max_support,
    )

    final_series: List[Optional[float]] = []
    stages: List[str] = []
    observed_count = 0
    small_gap_count = 0
    iterative_large_gap_count = 0

    for i, dp in enumerate(result):
        final_v = dp.imputed
        final_series.append(float(final_v) if final_v is not None else None)
        if raw_target[i] is not None:
            observed_count += 1
            stages.append("observed")
        elif target_series[i] is not None:
            small_gap_count += 1
            stages.append("small_gap_pchip")
        else:
            iterative_large_gap_count += 1
            stages.append("large_gap_mc_lnn_iterative")

    meta = {
        "Well_ID": wid,
        "n_months": len(months),
        "n_observed": observed_count,
        "n_small_gap_added": small_gap_count,
        "n_iterative_large_gap_added": iterative_large_gap_count,
        "n_unfilled": int(sum(v is None for v in final_series)),
        "outer_iterations_used": int(iter_meta.get("outer_iterations_used", 0)),
        "support_points": int(iter_meta.get("support_points", 0)),
        "best_support_kge": iter_meta.get("best_support_kge"),
        "best_support_rmse": iter_meta.get("best_support_rmse"),
    }
    return final_series, stages, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="PCHIP (small gaps) + MC-LNN (large gaps) imputer — matching aquiferx pipeline")
    parser.add_argument("--well-ids", type=str, default="", help="Comma-separated well IDs. Default: all eligible wells.")
    parser.add_argument("--output-tag", type=str, default="alleligible_iterative_softimpute_standalone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--feedback-prev-weight", type=float, default=0.35)
    parser.add_argument("--support-frac", type=float, default=0.12)
    parser.add_argument("--min-support", type=int, default=6)
    parser.add_argument("--max-support", type=int, default=24)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_module(APP_DIR / "run_gslb_long_gap_cv_mc_lnn.py", "mc_lnn_base")
    iterative = load_module(APP_DIR / "run_gslb_long_gap_cv_mc_lnn_iterative_softimpute.py", "mc_lnn_iter")

    params = base.build_params()
    main_df, aux_df, months, _top_wells, _donor_pool_wells, latlon = base.load_data()
    eligible_wells = sorted(latlon.keys())
    if args.well_ids.strip():
        requested = [w.strip() for w in args.well_ids.split(",") if w.strip()]
        available = set(eligible_wells)
        target_wells = [w for w in requested if w in available]
        missing = [w for w in requested if w not in available]
        if missing:
            print(f"Skipping unavailable wells: {missing}", flush=True)
        if not target_wells:
            raise ValueError("No requested wells were available after filtering.")
    else:
        target_wells = eligible_wells
        missing = []

    aux_lk = base.aux_lookup(aux_df)
    raw_obs = {wid: base.monthly_series(main_df, wid, months) for wid in eligible_wells}
    t0 = time.time()

    # ── PHASE 1: PCHIP interpolation (matching aquiferx pipeline) ──
    # PCHIP fills ALL gaps within the observation range, then large gaps
    # (> gap_size months) get blanked back to null (keeping pad at edges).
    # This matches the existing aquiferx interp_well() behavior.
    gap_size = 24  # months (~730 days, matching aquiferx default gap_size=730)
    pad_size = 6   # months (~180 days, matching aquiferx default pad=180)

    filled_obs: Dict[str, List[Optional[float]]] = {}
    for wid, raw in raw_obs.items():
        out = list(raw)
        obs_idx = [i for i, v in enumerate(raw) if v is not None]
        if len(obs_idx) < 3:
            filled_obs[wid] = out
            continue

        obs_vals = [raw[i] for i in obs_idx]
        first_obs = obs_idx[0]
        last_obs = obs_idx[-1]

        # Step 1: PCHIP fill everything within [first_obs, last_obs]
        try:
            interp = PchipInterpolator(obs_idx, obs_vals, extrapolate=False)
            for t in range(first_obs, last_obs + 1):
                if out[t] is None:
                    v = interp(t)
                    if np.isfinite(v):
                        out[t] = float(v)
        except Exception:
            pass

        # Step 2: Blank interior of large gaps (> gap_size), keep pad at edges
        # This leaves large gaps for MC-LNN to fill
        for g in range(len(obs_idx) - 1):
            gap_len = obs_idx[g + 1] - obs_idx[g] - 1
            if gap_len > gap_size:
                blank_start = obs_idx[g] + pad_size + 1
                blank_end = obs_idx[g + 1] - pad_size - 1
                for t in range(max(blank_start, 0), min(blank_end + 1, len(out))):
                    out[t] = None

        filled_obs[wid] = out

    series_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    total = len(target_wells)
    for idx, wid in enumerate(target_wells, start=1):
        full_series, stages, meta = impute_one_well(
            base=base,
            iterative=iterative,
            wid=wid,
            raw_target=list(raw_obs[wid]),
            target_series=list(filled_obs[wid]),
            donor_obs_filled=filled_obs,
            months=months,
            aux_lk=aux_lk,
            latlon=latlon,
            params=params,
            seed=base._rng_seed(args.seed, "iter_full", wid),
            outer_iterations=args.outer_iterations,
            feedback_prev_weight=args.feedback_prev_weight,
            support_frac=args.support_frac,
            min_support=args.min_support,
            max_support=args.max_support,
        )
        raw_series = raw_obs[wid]
        small_gap_series = filled_obs[wid]
        for m, raw_v, sg_v, final_v, stage in zip(months, raw_series, small_gap_series, full_series, stages):
            series_rows.append(
                {
                    "Well_ID": wid,
                    "Date": m.strftime("%Y-%m-%d"),
                    "raw_wte": raw_v,
                    "small_gap_wte": sg_v,
                    "final_wte": final_v,
                    "fill_stage": stage,
                }
            )
        summary_rows.append(meta)
        if idx % 25 == 0 or idx == total:
            print(f"[Standalone MC+LNN] {idx}/{total} wells complete in {time.time() - t0:.0f}s", flush=True)

    series_df = pd.DataFrame(series_rows)
    summary_df = pd.DataFrame(summary_rows)

    series_name = f"full_imputed_series_{args.output_tag}.csv"
    summary_name = f"full_imputed_summary_{args.output_tag}.csv"
    meta_name = f"run_metadata_{args.output_tag}.json"
    summary_txt_name = f"SUMMARY_{args.output_tag}.txt"

    series_df.to_csv(OUT_DIR / series_name, index=False)
    summary_df.to_csv(OUT_DIR / summary_name, index=False)

    run_meta = {
        "target_wells": target_wells,
        "missing_requested_wells": missing,
        "date_start": base.DATE_START,
        "date_end": base.DATE_END,
        "requested_mode": "small-gap PCHIPiliary, then large-gap iterative SoftImpute MC + LNN",
        "outer_iterations": args.outer_iterations,
        "feedback_prev_weight": args.feedback_prev_weight,
        "support_frac": args.support_frac,
        "min_support": args.min_support,
        "max_support": args.max_support,
        "params": asdict(params),
        "elapsed_seconds": time.time() - t0,
    }
    with open(OUT_DIR / meta_name, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    lines = [
        "Standalone GSLB full imputation summary",
        "Pipeline: small-gap PCHIPiliary -> large-gap iterative SoftImpute MC + LNN",
        f"Elapsed seconds: {run_meta['elapsed_seconds']:.1f}",
        f"Target wells: {len(target_wells)}",
        "",
        "Per-well counts:",
    ]
    for row in summary_rows[:50]:
        lines.append(
            f"  {row['Well_ID']}: observed={row['n_observed']}, "
            f"small-gap-added={row['n_small_gap_added']}, "
            f"iter-large-gap-added={row['n_iterative_large_gap_added']}, "
            f"unfilled={row['n_unfilled']}, "
            f"outer-iters-used={row['outer_iterations_used']}"
        )
    if len(summary_rows) > 50:
        lines.append(f"  ... {len(summary_rows) - 50} more wells omitted from text summary")
    lines.extend(
        [
            "",
            "Outputs:",
            f"  output/{series_name}",
            f"  output/{summary_name}",
            f"  output/{meta_name}",
        ]
    )
    with open(OUT_DIR / summary_txt_name, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
