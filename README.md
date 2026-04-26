# Aquifer Analyst

A web-based groundwater data visualization and analysis application built with React, TypeScript, and Vite. Import well data, explore interactive maps, run spatial interpolation and trend analyses, and fill data gaps with machine learning — all in the browser.

For full documentation, see [aquiferx.readthedocs.io](https://aquiferx.readthedocs.io/en/latest/).

## Quick Start

```bash
git clone https://github.com/njones61/aquiferx.git
cd aquiferx
npm install
npm run dev
```

## Bundled MC+LNN Imputer

This repository also bundles a standalone Python backend at `python/mc_lnn_imputer`.

Pipeline:
- `small-gap`: `LNN-CFC auxiliary`
- `large-gap`: iterative `SoftImpute MC -> LNN refinement`

Included inputs:
- `python/mc_lnn_imputer/datas/measurements_till_2023_to_lnn_imputation.csv`
- `python/mc_lnn_imputer/datas/lnn_imputation_gslb_gldas_df_excercise.csv`

Install Python dependencies:

```bash
npm run impute:mc-lnn:install
```

Run the standalone GSLB imputer:

```bash
npm run impute:mc-lnn:gslb
```

Outputs are written to `python/mc_lnn_imputer/output/`.

## License

MIT
