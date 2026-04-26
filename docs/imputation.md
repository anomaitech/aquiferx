# Imputing Data Gaps

## Bundled MC+LNN backend

AquiferX now ships with a standalone Python backend in `python/mc_lnn_imputer` for the Great Salt Lake Basin workflow. This is separate from the browser ELM pipeline described below. The bundled backend runs the validated two-stage process:

1. `small-gap` imputation with `LNN-CFC auxiliary`
2. `large-gap` imputation with iterative `SoftImpute MC -> LNN refinement`

The backend includes the required GSLB measurement and auxiliary CSV files and can run independently from the frontend:

```bash
npm run impute:mc-lnn:install
npm run impute:mc-lnn:gslb
```

Outputs are written to `python/mc_lnn_imputer/output/`.

Groundwater monitoring records are rarely complete. Wells get measured on irregular schedules, monitoring programs start and stop, funding changes, sensors fail, and the net result is a time series with gaps of months or years between contiguous runs of observations. For any analysis that benefits from continuous input — spatial interpolation at evenly-spaced frames, trend work over a uniform window, comparison across wells that happen to be measured on different schedules — those gaps have to be filled somehow. Aquifer Analyst's **Impute Data** workflow does this with a two-layer approach: a smooth interpolation through the measured intervals, and a machine-learning model that predicts values in the gaps and extrapolates beyond the last measurement.

The two layers handle different parts of the record. Within a well's measurement span, PCHIP interpolation produces monthly values that follow the actual observations closely. Outside that span — before the earliest measurement, after the latest, or across gaps longer than a configurable threshold — an Extreme Learning Machine (ELM) trained on soil moisture data from NASA's Global Land Data Assimilation System (GLDAS) fills in predictions based on climate signals that correlate with water table behavior at the well. The combined output is a continuous monthly series across your chosen date range, with the boundary between measurement-driven and model-driven regions smoothed so it doesn't jump visibly.

## Launching the Wizard

Click the **Impute Data** button in the toolbar to open the imputation wizard. The wizard has two steps: the first collects the output date range and the qualification criteria for wells; the second asks for a title and confirms the parameters before running.

<div style="color: #c00; background: #ffeaea; padding: 0.5em 0.75em; border-left: 4px solid #c00; margin: 1em 0;"><strong>SCREENSHOT NEEDED:</strong> Imputation wizard step 1 showing date range controls and well qualification filters</div>

### Choosing the output range and qualifying wells

The **Start Date** and **End Date** at the top of step 1 control the span of the output — the period over which the model will produce monthly predictions. Both dates are clipped to the GLDAS data availability window (approximately 1948 through the most recent GLDAS release, usually a month or two before present), since the model can't produce predictions outside the range where it has climate features to condition on. Plus and minus one-year buttons next to each date make quick adjustments easy.

Below the dates, the **Min Samples / Well** control sets the minimum number of actual measurements a well must have to be included in the imputation run. Wells with fewer than that many measurements are excluded — the ELM needs enough training data to fit, and wells with one or two observations don't give a regression anything to work with. The default of five is a practical minimum for reasonable model fits; raising it to ten or fifteen excludes more wells but produces tighter per-well R² values.

A real-time count of qualifying wells updates as you adjust the threshold, and a data density histogram below shows how many qualifying wells have measurements in each six-month bin across the output window. A bin that reads zero or near-zero is a stretch where even PCHIP won't have anything to interpolate from; seeing those gaps up front helps calibrate expectations for the output.

### Gap detection thresholds

Two parameters control the handoff between the PCHIP phase and the ELM phase. The **Gap Size (days)** threshold defines what counts as "too big to interpolate"; gaps in a well's measurement record that exceed this threshold are filled by the ELM model instead of by PCHIP. The default of 730 days (about two years) is a reasonable middle ground — shorter gaps are usually well-behaved for interpolation, while gaps of two or more years can span enough climatic variability that a pure interpolation is likely to miss real behavior.

The **Pad Size (days)** controls the transition zone at each edge of a large gap. PCHIP continues for this distance past the nearest measurement into the gap, and the ELM fills only the interior. The default of 180 days produces a smooth blend that avoids a visible step at the boundary; larger pads push more of the gap back to PCHIP, while smaller pads give the model more room to work but risk a more visible seam.

### Preview

A preview strip at the bottom of step 1 shows the PCHIP coverage for every qualifying well as a set of horizontal bars. Runs of filled color indicate periods where PCHIP can interpolate; gaps between the filled segments are what the ELM will fill. This is mostly a sanity check — if the preview shows widespread sparse coverage across all wells, either the minimum sample threshold is set too high or the aquifer simply doesn't have enough historical data for a productive imputation.

## The Second Step and the Run

Step 2 is short. Enter a title for the model — something descriptive like "Water Level 2024 Imputation" or "Nitrate ELM Run" — and review the summary of parameters below it. The title becomes the display label in the sidebar; the filename is derived from the title with a slug transformation.

<div style="color: #c00; background: #ffeaea; padding: 0.5em 0.75em; border-left: 4px solid #c00; margin: 1em 0;"><strong>SCREENSHOT NEEDED:</strong> Imputation wizard step 2 showing the title field, parameter summary, and log viewer during a run</div>

Clicking **Run** starts the imputation. A progress bar tracks the overall run, and a dark-terminal-style log viewer shows real-time processing messages — GLDAS fetch status, per-well training R² and RMSE values, warnings when a well fails to converge. Green log entries indicate successful training; red entries flag errors or poor fits. A typical run processes dozens of wells per minute, so an aquifer with a few hundred qualifying wells completes in a few minutes.

## How the Model Works

The imputation pipeline runs in three phases: a PCHIP interpolation pass over each well's measured data, a GLDAS fetch and feature assembly phase that builds the training matrix, and a per-well ELM training and prediction phase. The final output blends the first and third.

### The PCHIP phase

For each qualifying well, the pipeline fits a Piecewise Cubic Hermite Interpolating Polynomial through the well's measurements and samples it at monthly intervals between the earliest and latest measurement dates. PCHIP is monotonicity-preserving, which is the property that matters for groundwater data — it produces a smooth curve that doesn't overshoot between points the way an unconstrained cubic spline often does, so the interpolation respects the actual shape of the data rather than inventing peaks or troughs that aren't there.

Within the well's measurement span, PCHIP values are taken as-is for gaps smaller than the gap-size threshold. For gaps that exceed the threshold, PCHIP values are blanked out in the interior (minus the pad at each edge), leaving a region that will be filled by the ELM. The edges of every large gap retain PCHIP values out to the pad distance, so the model's output transitions smoothly from measurement-driven to climate-driven.

### GLDAS features

The ELM's input features come from NASA's Global Land Data Assimilation System (GLDAS), a monthly gridded climate and hydrology dataset with global coverage. For each aquifer, the pipeline fetches a monthly time series of soil moisture at the aquifer's centroid. Soil moisture is the single most informative GLDAS variable for unconfined water-table work — it correlates with recharge signals, runoff, and seasonal climate swings that translate (with lag) into water-table variability.

Five soil moisture features are computed: the raw monthly value, plus one-year, three-year, five-year, and ten-year rolling averages. The rolling averages capture different temporal scales. The one-year average smooths seasonal swings and reveals the underlying annual pattern; the three- and five-year averages pick up multi-year drought-and-recovery cycles; the ten-year average reflects longer climate oscillations and provides a slow-moving baseline. Each feature enters the model separately, letting the ELM learn how each time scale contributes at each well.

### Feature vector

For each month in the training window, the ELM sees a 19-element feature vector assembled from:

- Five z-scored soil moisture features (raw, plus the four rolling averages), z-scored using the global mean and standard deviation across the aquifer's full GLDAS time series.
- A min-max-scaled normalized year index that carries the overall temporal position.
- Twelve one-hot month indicators that let the model learn month-specific offsets independent of the climate signal.
- A constant 1.0 bias.

The one-hot month encoding is important because it lets the ELM learn that (say) February values are systematically different from August values at a particular well, independent of what the climate features are doing. This captures the phase relationship between climate drivers and the water table at each well — useful for wells where the response lags the climate signal by several months.

### The ELM itself

An Extreme Learning Machine is a single-hidden-layer neural network with an unusual training procedure. The input-to-hidden weights and hidden-layer biases are sampled randomly from a Gaussian distribution at the start of training and are never updated. Only the hidden-to-output weights are learned, and they're solved analytically via ridge regression rather than optimized iteratively.

Structurally, the network has 19 input neurons (one per feature), 500 hidden neurons with ReLU activations, and one output neuron that predicts the water level. The hidden layer activations are computed once per training point:

\[
\mathbf{H} = \text{ReLU}(\mathbf{X} \cdot \mathbf{W}_{in} + \mathbf{b})
\]

and the output weights come from the closed-form ridge-regression solution:

\[
\mathbf{W}_{out} = (\mathbf{H}^T \mathbf{H} + \lambda \mathbf{I})^{-1} \mathbf{H}^T \mathbf{y}
\]

with \(\lambda = 100\) acting as a regularization parameter that stabilizes the inversion and prevents overfitting. The whole training step is a single matrix operation, so an ELM fits orders of magnitude faster than a network trained with backpropagation. For the per-well setup — where the model is trained once per well, on at most a few hundred monthly training points — the speed is what makes the workflow practical.

### Per-well training

Each qualifying well gets its own independently-trained ELM. The pipeline uses that well's PCHIP-interpolated values (at the months where GLDAS data is available) as training targets, z-scoring them with the well's own mean and standard deviation so the model's output is on a normalized scale during training. The trained model then predicts the full GLDAS range — including the gaps and the extrapolation regions — and the predictions are un-normalized back to the original water-level scale.

Per-well rather than per-aquifer training is important because the relationship between soil moisture and water-table response varies from well to well. Wells in recharge areas respond quickly; wells in long-flow-path discharge areas respond slowly and out of phase. Wells under pumping stress may not correlate with climate drivers at all. Training independently lets each well's model discover whatever relationship actually applies at that location, rather than forcing a single aquifer-wide fit that averages over heterogeneous behavior.

### Combining the two

The output for each well and month is the PCHIP value if one exists (measurement-supported periods, plus the pad zones at the edges of large gaps) and the ELM prediction otherwise (gap interiors, and any extrapolation before or after the measurement span). The combined series is what gets stored and what appears when you load the model from the sidebar.

## Viewing the Results

Completed imputation models appear in the sidebar under the aquifer they were run for. Clicking the model loads it — the chart switches to show each selected well's combined time series, the legend shows the well names, and two quality metrics appear as badges in the chart header.

<div style="color: #c00; background: #ffeaea; padding: 0.5em 0.75em; border-left: 4px solid #c00; margin: 1em 0;"><strong>SCREENSHOT NEEDED:</strong> Model chart showing combined PCHIP+ELM curves with R² and RMSE badges</div>

The chart offers two display modes. **Combined mode** — the default — draws the PCHIP portion in red and the ELM portion in blue on a single continuous curve. The two colors make it visually obvious which parts of the series are supported by measurements and which are model predictions; a well with lots of red and a short blue segment at the start is mostly measurement-driven, while a well with a large blue fill in the middle is substantially model-driven.

**Uncombined mode** overlays both a PCHIP curve and an ELM prediction across the full range of each, with the actual measurement points as green dots. This is useful for verifying model quality — when ELM predictions track closely with PCHIP over periods where both are available, the model has learned something sensible from the climate features; where the two diverge, the model's predictions for the gap regions are worth scrutinizing.

### Quality metrics

Two metrics appear as badges in the chart header when a model well is selected. **R²** is the coefficient of determination, measuring how much of the variance in the training targets the model explains. Values above 0.7 typically indicate a good climate-driven fit; values below 0.5 suggest that soil moisture alone doesn't explain the well's behavior well. **RMSE** is the root-mean-squared error in the region's length unit (feet or meters), representing the average prediction error magnitude. Both metrics are computed on the training set in the well's original (denormalized) scale.

Wells with low R² are not necessarily bugs in the imputation — they're signals about the well's physical behavior. A pumping well under active irrigation may show almost no climate correlation. A well in a confined aquifer may lag seasonal signals by years. A well with a mix of pumping and recharge-driven behavior may have a genuinely low climate-driven signal that the model can't improve on. The processing log preserves per-well R² and RMSE values, which is useful for flagging which wells in the run are reliable model predictions and which are closer to the PCHIP envelope with climate-shaped extrapolation.

### Smoothing overlay

The chart's **Smooth** toggle applies a Nadaraya-Watson Gaussian kernel smoothing to the combined output, with a configurable window in months. This is useful when the combined curve has month-to-month wiggles driven by the climate features that you don't care about for the analysis at hand. The smoothed curve appears as an orange dashed overlay; a twelve-month window is a reasonable default for de-seasoning, while longer windows emphasize multi-year trends.

## Using Imputation Output Downstream

An imputation model can feed a spatial analysis. When you launch the spatial analysis wizard on an aquifer that has a completed model, the temporal-method selector offers a "Model" option that pulls the combined imputation output at each well instead of interpolating on-the-fly from the raw measurements. This produces raster surfaces that benefit from the gap-filled record — areas where measurements are sparse in time don't produce spatial holes at animation frames.

The caveat is that the raster surface inherits the model's uncertainty. A frame drawn entirely from ELM predictions is as good as the ELM predictions themselves, which is to say, only as reliable as the climate-to-water-level relationship at each well. The R² badges on the chart and in the processing log are the main diagnostic for deciding whether a given model is appropriate to use for spatial work.

## Tips

A few patterns worth knowing:

**Low R² at most wells** often means soil moisture isn't the dominant driver for the aquifer — pumping, irrigation return flow, or confined-aquifer isolation may be running the show. The model will still produce estimates but they shouldn't be over-interpreted. Consider whether a different temporal smoothing (a simple PCHIP with short-gap bridging) might serve the downstream analysis as well.

**A few wells with very high RMSE** can usually be traced to either unusual individual wells (wells with measurements in odd units, wells near an active pumping center) or to the minimum-sample threshold being too low for the aquifer's measurement density. Bumping the minimum to ten or fifteen samples typically weeds out the worst fits.

**Gap size** is the main lever for the PCHIP-vs-ELM trade-off. Lowering it pushes more of the record to PCHIP, which follows measurements closely but extrapolates poorly. Raising it gives the model more room to work and produces a more uniform set of predictions across wells, at the cost of some measurement fidelity in longer gaps.

**Pad size** controls the visual transition. If you see visible seams where PCHIP meets ELM in the chart, raising the pad to 360 days or more smooths the join at the cost of shrinking the ELM's active region.
