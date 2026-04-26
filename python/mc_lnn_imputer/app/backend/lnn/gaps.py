"""
Gap detection and outlier identification.
Mirrors utils/math.ts: gap logic in batchImputeSmallGapInstances, identifyOutliers (z-score > 2.5).
"""

from typing import List, Dict, Set, Tuple

from .types import DataPoint
from .math_utils import calculate_mean, calculate_std_dev


def identify_gaps(
    data: List[DataPoint],
    max_gap_threshold: int,
) -> Tuple[List[Dict], Set[int]]:
    """
    Find all gaps (consecutive missing observed) and mark large-gap indices.
    JS: same logic in batchImputeSmallGapInstances (gaps array, largeGapIndices set).
    Returns:
        gaps: list of {"startIdx", "endIdx", "size"}
        large_gap_indices: set of indices that belong to gaps with size > max_gap_threshold
    """
    gaps: List[Dict] = []
    large_gap_indices: Set[int] = set()
    current_gap = 0
    gap_start_idx = -1
    seen_observed = False

    for idx, d in enumerate(data):
        if d.observed is None:
            if not seen_observed:
                # Leading missing run is extrapolation, not an interior gap.
                continue
            if current_gap == 0:
                gap_start_idx = idx
            current_gap += 1
        else:
            seen_observed = True
            if current_gap > 0:
                gaps.append({
                    "startIdx": gap_start_idx,
                    "endIdx": idx - 1,
                    "size": current_gap,
                })
                if current_gap > max_gap_threshold:
                    for i in range(gap_start_idx, idx):
                        large_gap_indices.add(i)
                current_gap = 0

    # Trailing missing run is extrapolation, not an interior gap, so it is ignored.

    return gaps, large_gap_indices


def identify_outliers(data: List[DataPoint]) -> List[int]:
    """
    Outliers = points with imputed value whose z-score (vs mean/std of imputed values) > 2.5.
    JS: identifyOutliers (only considers points that have imputed values).
    Returns list of indices into data.
    """
    valid_points = [(i, d) for i, d in enumerate(data) if d.imputed is not None]
    if len(valid_points) < 5:
        return []
    imputed_values = [d.imputed for _, d in valid_points]
    mean_imp = calculate_mean(imputed_values)
    std_imp = calculate_std_dev(imputed_values, mean_imp)
    if std_imp == 0:
        return []
    outlier_indices: List[int] = []
    for i, d in valid_points:
        z_score = (d.imputed - mean_imp) / std_imp
        if abs(z_score) > 2.5:
            outlier_indices.append(i)
    return outlier_indices
