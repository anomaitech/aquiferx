# Browser MC+LNN Port

Goal: move the validated GSLB imputation pipeline from the bundled Python sidecar into a browser-native AquiferX implementation.

## Validated reference pipeline

1. `small-gap` imputation with `LNN-CFC auxiliary`
2. `large-gap` iterative `SoftImpute MC -> LNN refinement`

## Port order

1. Browser-safe `SoftImpute` / matrix completion core
2. Shared time-grid and feature assembly
3. `LNN-CFC auxiliary` TypeScript port for small gaps
4. Iterative `MC -> LNN -> MC -> LNN` feedback loop
5. Web Worker execution path to keep the UI responsive
6. Wire results into the existing AquiferX imputation model format

## Why this order

- The MC core already outperformed IDW in held-out interpolation tests.
- `SoftImpute` is deterministic, compact, and maps cleanly to `ml-matrix`.
- The LNN port is the harder part and should be layered on top of a working browser MC engine, not attempted blind.

## Current foundation

- `services/mcSoftImpute.ts`
- `services/mcLnnBrowser.ts`

These files establish the browser-native large-gap MC core without misrepresenting the incomplete LNN port as finished.
