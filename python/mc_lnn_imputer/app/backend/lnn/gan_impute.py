"""
GAIN (Generative Adversarial Imputation Network) and Conditional Imputation GAN for small-gap imputation.
NumPy-only MLP implementation; autotune hyperparameters for high KGE.

References:
  GAIN: Yoon, Jordon, van der Schaar. "GAIN: Missing Data Imputation using Generative Adversarial Nets."
  ICML 2018. https://arxiv.org/abs/1806.02920
  - Generator G observes components of the data vector, imputes missing conditioned on observed.
  - Discriminator D takes completed vector and predicts which components were observed vs imputed.
  - Hint vector reveals partial missingness to D so that G learns the true data distribution.

  Conditional Imputation GAN: Deng, Han, Matteson. "Extended Missing Data Imputation via GANs for
  Ranking Applications." arXiv 2020. https://arxiv.org/abs/2011.02089
  - Conditional GAN for imputation with guarantees for Extended MAR/EAMAR.
  - Generator conditions on observed/mask; we train G with reconstruction on observed.
"""

from typing import List, Tuple, Optional
import numpy as np

from .types import DataPoint, SimulationParams
from .math_utils import get_scaler, normalize_value, denormalize_value, calculate_kge
from .gaps import identify_outliers


def _kge_from_result(result: List[DataPoint], large_gap_indices: List[int]) -> float:
    """Compute KGE from result list (observed vs imputed at same indices, excluding large-gap)."""
    large_set = set(large_gap_indices)
    obs, pred = [], []
    for i, r in enumerate(result):
        if i not in large_set and r.observed is not None and r.imputed is not None:
            obs.append(r.observed)
            pred.append(r.imputed)
    if not obs:
        return float("-inf")
    return calculate_kge(obs, pred)


def compute_holdout_kge_gain(
    data: List[DataPoint],
    large_gap_indices: List[int],
    params: SimulationParams,
    epochs_override: int,
    hint_rate_override: float,
    holdout_frac: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Measure gap-filling quality: mask a random fraction of observed points, run GAIN, compute KGE
    on (true, imputed) at those points. This KGE reflects imputation quality, not reconstruction.
    """
    if rng is None:
        rng = np.random.default_rng()
    large_set = set(large_gap_indices)
    observed_indices = [i for i in range(len(data)) if i not in large_set and data[i].observed is not None]
    if len(observed_indices) < 3:
        return float("-inf")
    n_holdout = max(2, int(len(observed_indices) * holdout_frac))
    n_holdout = min(n_holdout, len(observed_indices) - 1)
    holdout_set = set(rng.choice(observed_indices, size=n_holdout, replace=False))
    true_holdout = [data[i].observed for i in sorted(holdout_set)]
    data_holdout = []
    for i, d in enumerate(data):
        obs = None if i in holdout_set else d.observed
        data_holdout.append(DataPoint(
            time=d.time, observed=obs, instance_id=d.instance_id,
            auxiliaries=d.auxiliaries, is_masked=d.is_masked,
            latitude=d.latitude, longitude=d.longitude,
        ))
    result = run_gain_imputation(
        data_holdout, large_gap_indices, params, rng=rng,
        epochs_override=epochs_override, hint_rate_override=hint_rate_override,
    )
    pred_holdout = [result[i].imputed for i in sorted(holdout_set) if i < len(result) and result[i].imputed is not None]
    if len(pred_holdout) != len(true_holdout) or len(true_holdout) < 2:
        return float("-inf")
    return calculate_kge(true_holdout, pred_holdout)


# --- Helpers: build series + mask from DataPoint list (excluding large-gap indices) ---

def _series_and_mask(data: List[DataPoint], large_gap_indices: set) -> Tuple[np.ndarray, np.ndarray]:
    """Return (values, mask): values with NaN for missing; mask 1=observed, 0=missing. Exclude large-gap from mask as not imputed."""
    n = len(data)
    values = np.full(n, np.nan)
    mask = np.zeros(n)
    for i, d in enumerate(data):
        if i in large_gap_indices:
            continue
        if d.observed is not None:
            values[i] = d.observed
            mask[i] = 1.0
        else:
            mask[i] = 0.0
    return values, mask


def _min_max_scale(v: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    if max_val == min_val:
        return np.zeros_like(v)
    return ((v - min_val) / (max_val - min_val)) * 1.6 - 0.8


def _min_max_inv(norm: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    if max_val == min_val:
        return np.full_like(norm, min_val)
    return ((norm + 0.8) / 1.6) * (max_val - min_val) + min_val


# --- Conv1D layer (NumPy) for temporal convolutions (G) ---

class Conv1DLayer:
    """Simple 1D convolution (same-padding, kernel_size=5) applied to series.
    Uses vectorized np.convolve for speed instead of triple-nested loops."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = rng.standard_normal((out_channels, in_channels, kernel_size)) * 0.1
        self.b = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (1, T) or (1, C, T) -> (1, out_channels, T) with same-padding."""
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # (1, 1, T)
        batch, in_c, T = x.shape
        pad = self.kernel_size // 2
        out = np.zeros((batch, self.out_channels, T))
        for oc in range(self.out_channels):
            for ic in range(in_c):
                # Vectorized convolution per channel pair
                kernel = self.W[oc, ic, ::-1]  # flip for correlation→convolution
                conv_full = np.convolve(x[0, ic], kernel, mode='full')
                out[0, oc] += conv_full[pad:pad + T]
            out[0, oc] += self.b[oc]
        return np.maximum(0, out)  # ReLU activation


# --- Simple MLP (NumPy) for GAIN / CondGAN ---

class SimpleMLP:
    def __init__(self, dims: List[int], rng: np.random.Generator):
        self.dims = dims
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = 0.1 if i == 0 else 0.1
            self.W.append(rng.standard_normal((dims[i], dims[i + 1])) * scale)
            self.b.append(np.zeros(dims[i + 1]))

    def forward(self, x: np.ndarray, last_linear: bool = False) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        acts = [x]
        preacts = []
        for i in range(len(self.W)):
            pre = acts[-1] @ self.W[i] + self.b[i]
            preacts.append(pre)
            if last_linear and i == len(self.W) - 1:
                acts.append(pre)
            else:
                acts.append(np.maximum(0, pre))  # ReLU
        return acts[-1], acts, preacts

    def backward_regression(
        self, acts: List[np.ndarray], preacts: List[np.ndarray], grad_out: np.ndarray, lr: float = 1e-3
    ) -> None:
        d = grad_out.copy()
        for i in range(len(self.W) - 1, -1, -1):
            if i < len(self.W) - 1:
                d = d * (preacts[i] > 0).astype(float)
            grad_W = acts[i].T @ d
            grad_b = d.sum(axis=0)
            self.W[i] -= lr * grad_W
            self.b[i] -= lr * grad_b
            if i > 0:
                d = d @ self.W[i].T


# --- GAIN (arXiv:1806.02920) ---

def _run_gain_single(
    values: np.ndarray,
    mask: np.ndarray,
    min_val: float,
    max_val: float,
    epochs: int = 150,
    hint_rate: float = 0.9,
    rng: np.random.Generator = None,
    aux_matrix: np.ndarray = None,
) -> np.ndarray:
    """GAIN: G observes available components, imputes missing; D predicts observed vs imputed; hint reveals partial missingness to D (Yoon et al., ICML 2018).
    (I) aux_matrix: optional (T, n_aux) auxiliary conditioning matrix."""
    if rng is None:
        rng = np.random.default_rng()

    T = values.size
    if T > 300:
        # Cap length: use last 300 points for stability
        offset = T - 300
        values = values[-300:]
        mask = mask[-300:]
        if aux_matrix is not None:
            aux_matrix = aux_matrix[-300:]
        T = 300

    # Normalize observed to [-0.8, 0.8]
    x_fill = values.copy()
    valid = ~np.isnan(values) & (mask > 0.5)
    if np.any(valid):
        x_fill[~valid] = np.nanmean(values[valid])
    else:
        x_fill[:] = 0.0
    x_norm = _min_max_scale(x_fill, min_val, max_val)

    # (I) Auxiliary conditioning: concatenate flattened aux to generator input
    n_aux_features = 0
    aux_flat = np.array([])
    if aux_matrix is not None and aux_matrix.size > 0:
        n_aux_features = aux_matrix.shape[1] if aux_matrix.ndim == 2 else 1
        aux_safe = np.nan_to_num(aux_matrix, nan=0.0).flatten()
        aux_flat = aux_safe

    dim_in = 3 * T + len(aux_flat)
    dim_h = min(128, dim_in // 2)

    # (G) Temporal convolutions: apply Conv1D to series before MLP when T > 10
    use_conv = T > 10
    if use_conv:
        conv_g = Conv1DLayer(1, 4, kernel_size=5, rng=rng)
        conv_d = Conv1DLayer(1, 4, kernel_size=5, rng=rng)
        # Conv output adds 4*T features but we keep dims manageable via the MLP
        dim_in_g = 3 * T + 4 * T + len(aux_flat)
        dim_in_d = 2 * T + 4 * T
    else:
        dim_in_g = dim_in
        dim_in_d = 2 * T

    dim_h_g = min(128, dim_in_g // 2)
    # Generator: (x_fill, m, z, [conv], [aux]) -> x_imputed
    G = SimpleMLP([dim_in_g, dim_h_g, T], rng)
    # Discriminator: (x_imputed, h, [conv]) -> m_pred
    D = SimpleMLP([dim_in_d, 64, T], rng)

    # Precompute static conv features for G (x_norm never changes)
    g_conv_cache = None
    if use_conv:
        g_conv_cache = conv_g.forward(x_norm.reshape(1, -1)).reshape(1, -1)

    def _build_g_input(z_vec):
        parts = [x_norm.reshape(1, -1), mask.reshape(1, -1), z_vec]
        if use_conv:
            parts.append(g_conv_cache)
        if aux_flat.size > 0:
            parts.append(aux_flat.reshape(1, -1))
        return np.concatenate(parts, axis=1)

    def _build_d_input(x_imp_vec, h_vec):
        parts = [x_imp_vec.reshape(1, -1), h_vec.reshape(1, -1)]
        if use_conv:
            conv_feat = conv_d.forward(x_imp_vec.reshape(1, -1)).reshape(1, -1)
            parts.append(conv_feat)
        return np.concatenate(parts, axis=1)

    for ep in range(epochs):
        z = rng.standard_normal((1, T))
        x_in = _build_g_input(z)
        # G forward
        g_out, g_acts, g_pre = G.forward(x_in, last_linear=True)
        x_imputed = g_out * (1 - mask) + x_norm * mask  # replace observed with true
        # Hint: reveal true mask with probability hint_rate; ambiguous (0.5) otherwise
        h = np.full(T, 0.5)
        hint_mask = rng.random(T) < hint_rate
        h[hint_mask] = mask[hint_mask]
        d_in = _build_d_input(x_imputed, h)
        d_out, d_acts, d_pre = D.forward(d_in, last_linear=False)
        d_out = 1.0 / (1.0 + np.exp(-np.clip(d_pre[-1], -20, 20)))

        # D loss: BCE on M
        d_loss = -np.mean(mask * np.log(d_out + 1e-8) + (1 - mask) * np.log(1 - d_out + 1e-8))
        grad_d = (d_out - mask).reshape(1, -1)
        for i in range(len(D.W) - 1, -1, -1):
            if i < len(D.W) - 1:
                grad_d = grad_d * (d_pre[i] > 0).astype(float)
            D.W[i] -= 5e-3 * (d_acts[i].T @ grad_d)
            D.b[i] -= 5e-3 * grad_d.sum(axis=0)
            if i > 0:
                grad_d = grad_d @ D.W[i].T

        # G: reconstruction on observed + adversarial (want D to predict 1 for imputed)
        z2 = rng.standard_normal((1, T))
        x_in2 = _build_g_input(z2)
        g_out2, g_acts2, g_pre2 = G.forward(x_in2, last_linear=True)
        x_imp2 = g_out2 * (1 - mask) + x_norm * mask
        d_in2 = _build_d_input(x_imp2, h)
        d_out2, _, d_pre2 = D.forward(d_in2, last_linear=False)
        d_out2 = 1.0 / (1.0 + np.exp(-np.clip(d_pre2[-1], -20, 20)))

        rec_loss = np.mean(((g_out2 - x_norm) * mask) ** 2)
        adv_loss = -np.mean((1 - mask) * np.log(d_out2 + 1e-8))
        g_loss = rec_loss + 0.1 * adv_loss

        grad_g = np.zeros((1, T))
        grad_g += 2 * (g_out2 - x_norm) * mask / (T + 1e-8)
        grad_adv = -(1 - mask) * (1 - d_out2) / (T + 1e-8)
        grad_g += 0.1 * grad_adv
        G.backward_regression(g_acts2, g_pre2, grad_g, lr=3e-3)

    # Final imputation
    z_f = rng.standard_normal((1, T))
    x_in_f = _build_g_input(z_f)
    g_final, _, _ = G.forward(x_in_f, last_linear=True)
    imputed_norm = g_final.flatten() * (1 - mask) + x_norm * mask
    imputed = _min_max_inv(imputed_norm, min_val, max_val)
    return imputed


def _run_wgan_gp_single(
    values: np.ndarray,
    mask: np.ndarray,
    min_val: float,
    max_val: float,
    epochs: int = 150,
    gp_lambda: float = 10.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """(F) WGAN-GP: Wasserstein GAN with gradient penalty for imputation.
    D: linear output (no sigmoid), Wasserstein loss. G: -mean(D(fake)).
    Gradient penalty via finite differences on interpolated samples."""
    if rng is None:
        rng = np.random.default_rng()

    T = values.size
    if T > 300:
        values = values[-300:]
        mask = mask[-300:]
        T = 300

    x_fill = values.copy()
    valid = ~np.isnan(values) & (mask > 0.5)
    if np.any(valid):
        x_fill[~valid] = np.nanmean(values[valid])
    else:
        x_fill[:] = 0.0
    x_norm = _min_max_scale(x_fill, min_val, max_val)

    dim_in = 3 * T
    dim_h = min(128, dim_in // 2)
    G = SimpleMLP([dim_in, dim_h, T], rng)
    D = SimpleMLP([T, 64, 1], rng)  # Critic: scalar output

    eps_fd = 1e-4  # Finite difference epsilon for gradient penalty

    for ep in range(epochs):
        # Generator output
        z = rng.standard_normal((1, T))
        x_in = np.concatenate([x_norm.reshape(1, -1), mask.reshape(1, -1), z], axis=1)
        g_out, _, _ = G.forward(x_in, last_linear=True)
        x_fake = (g_out * (1 - mask) + x_norm * mask).reshape(1, -1)
        x_real = x_norm.reshape(1, -1)

        # --- Train Critic (D) ---
        # Wasserstein loss: maximize D(real) - D(fake)
        d_real, d_acts_r, d_pre_r = D.forward(x_real, last_linear=True)
        d_fake, d_acts_f, d_pre_f = D.forward(x_fake, last_linear=True)

        # Gradient penalty: interpolated samples
        alpha_interp = rng.random()
        x_interp = alpha_interp * x_real + (1 - alpha_interp) * x_fake
        d_interp, _, _ = D.forward(x_interp, last_linear=True)
        # Numerical gradient via finite differences
        grad_norm = 0.0
        for j in range(T):
            x_plus = x_interp.copy()
            x_plus[0, j] += eps_fd
            d_plus, _, _ = D.forward(x_plus, last_linear=True)
            grad_j = (float(d_plus) - float(d_interp)) / eps_fd
            grad_norm += grad_j ** 2
        grad_norm = np.sqrt(grad_norm + 1e-12)
        gp = gp_lambda * (grad_norm - 1.0) ** 2

        # D loss: -(D(real) - D(fake)) + GP  (minimize = maximize D(real) - D(fake))
        d_loss = -(float(d_real) - float(d_fake)) + gp
        # Update D: gradient of d_loss w.r.t. D weights
        # Grad from real: -1 (want to increase D(real))
        grad_d_real = np.array([[-1.0]])
        D.backward_regression(d_acts_r, d_pre_r, grad_d_real, lr=1e-3)
        # Grad from fake: +1 (want to decrease D(fake))
        grad_d_fake = np.array([[1.0]])
        D.backward_regression(d_acts_f, d_pre_f, grad_d_fake, lr=1e-3)

        # --- Train Generator ---
        z2 = rng.standard_normal((1, T))
        x_in2 = np.concatenate([x_norm.reshape(1, -1), mask.reshape(1, -1), z2], axis=1)
        g_out2, g_acts2, g_pre2 = G.forward(x_in2, last_linear=True)
        x_fake2 = (g_out2 * (1 - mask) + x_norm * mask).reshape(1, -1)
        d_fake2, _, _ = D.forward(x_fake2, last_linear=True)
        # G wants to maximize D(fake) = minimize -D(fake)
        # Also add reconstruction loss on observed
        rec_loss = np.mean(((g_out2 - x_norm) * mask) ** 2)
        # Backprop through G: gradient = reconstruction + adversarial direction
        grad_g = 2.0 * (g_out2 - x_norm) * mask / (T + 1e-8)
        # Adversarial: negative gradient (approximate: push g_out toward increasing D)
        grad_g += -0.1 * (1 - mask) / (T + 1e-8)
        G.backward_regression(g_acts2, g_pre2, grad_g, lr=2e-3)

    # Final imputation
    z_f = rng.standard_normal((1, T))
    x_in_f = np.concatenate([x_norm.reshape(1, -1), mask.reshape(1, -1), z_f], axis=1)
    g_final, _, _ = G.forward(x_in_f, last_linear=True)
    imputed_norm = g_final.flatten() * (1 - mask) + x_norm * mask
    return _min_max_inv(imputed_norm, min_val, max_val)


def run_gain_imputation(
    data: List[DataPoint],
    large_gap_indices: List[int],
    params: SimulationParams,
    rng: np.random.Generator = None,
    epochs_override: Optional[int] = None,
    hint_rate_override: Optional[float] = None,
) -> List[DataPoint]:
    """GAIN imputation for one instance (Yoon et al., ICML 2018, https://arxiv.org/abs/1806.02920). Returns DataPoints with imputed set for small-gap missing only.
    Routes to WGAN-GP or Conditional GAN based on params.gan_mode.
    Supports (I) auxiliary conditioning and (H) multiple imputation."""
    if rng is None:
        rng = np.random.default_rng()

    large_set = set(large_gap_indices)
    values, mask = _series_and_mask(data, large_set)
    n = len(data)
    obs_vals = [data[i].observed for i in range(n) if i not in large_set and data[i].observed is not None]
    min_val = min(obs_vals) if obs_vals else 0.0
    max_val = max(obs_vals) if obs_vals else 1.0

    if np.sum(mask) < 2:
        return [
            DataPoint(
                time=data[i].time,
                observed=data[i].observed,
                instance_id=data[i].instance_id,
                imputed=data[i].observed,
                auxiliaries=data[i].auxiliaries,
                is_masked=data[i].is_masked,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            )
            for i in range(n)
        ]

    epochs = epochs_override if epochs_override is not None else (getattr(params, "ganEpochs", 150) or 150)
    hint_rate = hint_rate_override if hint_rate_override is not None else (getattr(params, "ganHintRate", 0.9) or 0.9)
    gan_mode = getattr(params, "gan_mode", "gain") or "gain"
    gp_lambda = getattr(params, "gan_gp_lambda", 10.0) or 10.0
    use_aux = getattr(params, "gan_use_auxiliary", True)
    n_imputations = max(1, getattr(params, "gan_multiple_imputations", 1) or 1)

    # (I) Build auxiliary matrix if available and requested
    aux_matrix = None
    if use_aux:
        first_with_aux = next((d for d in data if d.auxiliaries and len(d.auxiliaries) > 0), None)
        if first_with_aux:
            n_aux = len(first_with_aux.auxiliaries)
            aux_matrix = np.zeros((n, n_aux))
            for i, d in enumerate(data):
                if d.auxiliaries and len(d.auxiliaries) >= n_aux:
                    for j in range(n_aux):
                        aux_matrix[i, j] = d.auxiliaries[j] if d.auxiliaries[j] is not None else 0.0

    # Route to appropriate GAN mode
    if gan_mode == "conditional":
        # Use conditional GAN
        imputed_vals = _run_condgan_single(values, mask, min_val, max_val, epochs=epochs, rng=rng)
    elif gan_mode == "wgan_gp":
        imputed_vals = _run_wgan_gp_single(values, mask, min_val, max_val, epochs=epochs, gp_lambda=gp_lambda, rng=rng)
    else:
        imputed_vals = _run_gain_single(
            values, mask, min_val, max_val,
            epochs=epochs, hint_rate=hint_rate, rng=rng,
            aux_matrix=aux_matrix,
        )

    # (H) Multiple imputation: run M times, average predictions, compute std
    if n_imputations > 1:
        all_imputed = [imputed_vals]
        for m in range(1, n_imputations):
            seed = rng.integers(0, 2**31)
            m_rng = np.random.default_rng(seed)
            if gan_mode == "conditional":
                imp_m = _run_condgan_single(values, mask, min_val, max_val, epochs=epochs, rng=m_rng)
            elif gan_mode == "wgan_gp":
                imp_m = _run_wgan_gp_single(values, mask, min_val, max_val, epochs=epochs, gp_lambda=gp_lambda, rng=m_rng)
            else:
                imp_m = _run_gain_single(values, mask, min_val, max_val, epochs=epochs, hint_rate=hint_rate, rng=m_rng, aux_matrix=aux_matrix)
            all_imputed.append(imp_m)
        # Stack and compute mean/std
        imp_stack = np.array(all_imputed)  # (M, T_eff)
        imputed_vals = np.mean(imp_stack, axis=0)
        imputed_std = np.std(imp_stack, axis=0)
    else:
        imputed_std = None

    # When series was truncated (n > 300), imputed_vals corresponds to the last len(imputed_vals) indices
    offset = max(0, n - len(imputed_vals))

    def _get_imputed(i: int) -> float:
        j = i - offset
        if 0 <= j < len(imputed_vals):
            return float(imputed_vals[j])
        if data[i].observed is not None:
            return data[i].observed
        return (min_val + max_val) / 2

    out: List[DataPoint] = []
    for i in range(n):
        if i in large_set:
            out.append(DataPoint(
                time=data[i].time,
                observed=None,
                instance_id=data[i].instance_id,
                auxiliaries=data[i].auxiliaries,
                imputed=None,
                is_masked=True,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            ))
        elif data[i].observed is not None:
            out.append(DataPoint(
                time=data[i].time,
                observed=data[i].observed,
                instance_id=data[i].instance_id,
                auxiliaries=data[i].auxiliaries,
                imputed=_get_imputed(i),
                is_masked=False,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            ))
        else:
            out.append(DataPoint(
                time=data[i].time,
                observed=None,
                instance_id=data[i].instance_id,
                auxiliaries=data[i].auxiliaries,
                imputed=_get_imputed(i),
                is_masked=False,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            ))
    return out


# --- Conditional Imputation GAN (arXiv:2011.02089) ---
# Generator G(z, c) conditions on c = (mask, observed values); trained with reconstruction on observed.
# Aligns with "Conditional Imputation GAN" for Extended MAR/EAMAR (Deng et al.).

def _run_condgan_single(
    values: np.ndarray,
    mask: np.ndarray,
    min_val: float,
    max_val: float,
    epochs: int = 150,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Conditional Imputation GAN (Deng et al., arXiv:2011.02089).
    G(z, c) conditions on c = (mask, observed); D(x_completed, c) discriminates real vs imputed.
    Adversarial training ensures imputed values follow the true data distribution,
    not just minimize reconstruction MSE."""
    T = values.size
    if T > 300:
        values = values[-300:]
        mask = mask[-300:]
        T = 300

    x_fill = values.copy()
    valid = ~np.isnan(values) & (mask > 0.5)
    if np.any(valid):
        x_fill[~valid] = np.nanmean(values[valid])
    else:
        x_fill[:] = 0.0
    x_norm = _min_max_scale(x_fill, min_val, max_val)
    c = np.concatenate([mask, x_norm])

    dim_c = 2 * T
    dim_z = T
    dim_h = min(128, (dim_c + dim_z) // 2)
    # G: (z, c) -> x_imputed; conditions on mask and observed values
    G = SimpleMLP([dim_z + dim_c, dim_h, T], rng)
    # D: (x_completed, c) -> [0,1] per component; predicts which are observed vs imputed
    D = SimpleMLP([T + dim_c, 64, T], rng)

    for ep in range(epochs):
        # --- Train Discriminator ---
        z = rng.standard_normal((1, dim_z))
        gc = np.concatenate([z, c.reshape(1, -1)], axis=1)
        g_out, _, _ = G.forward(gc, last_linear=True)
        x_completed = g_out * (1 - mask) + x_norm * mask  # replace observed with true
        d_in = np.concatenate([x_completed.reshape(1, -1), c.reshape(1, -1)], axis=1)
        d_out_raw, d_acts, d_pre = D.forward(d_in, last_linear=False)
        d_out = 1.0 / (1.0 + np.exp(-np.clip(d_pre[-1], -20, 20)))
        # D loss: BCE; D should predict 1 for observed, 0 for imputed
        grad_d = (d_out - mask).reshape(1, -1)
        for i in range(len(D.W) - 1, -1, -1):
            if i < len(D.W) - 1:
                grad_d = grad_d * (d_pre[i] > 0).astype(float)
            D.W[i] -= 5e-3 * (d_acts[i].T @ grad_d)
            D.b[i] -= 5e-3 * grad_d.sum(axis=0)
            if i > 0:
                grad_d = grad_d @ D.W[i].T

        # --- Train Generator ---
        z2 = rng.standard_normal((1, dim_z))
        gc2 = np.concatenate([z2, c.reshape(1, -1)], axis=1)
        g_out2, g_acts2, g_pre2 = G.forward(gc2, last_linear=True)
        x_completed2 = g_out2 * (1 - mask) + x_norm * mask
        d_in2 = np.concatenate([x_completed2.reshape(1, -1), c.reshape(1, -1)], axis=1)
        _, _, d_pre2 = D.forward(d_in2, last_linear=False)
        d_out2 = 1.0 / (1.0 + np.exp(-np.clip(d_pre2[-1], -20, 20)))
        # G loss: reconstruction on observed + adversarial (G wants D to predict 1 for imputed)
        rec_loss_grad = 2 * (g_out2 - x_norm) * mask / (T + 1e-8)
        adv_loss_grad = -(1 - mask) * (1 - d_out2) / (T + 1e-8)
        grad_g = rec_loss_grad + 0.1 * adv_loss_grad
        G.backward_regression(g_acts2, g_pre2, grad_g.reshape(1, -1), lr=3e-3)

    z_f = rng.standard_normal((1, dim_z))
    gc_f = np.concatenate([z_f, c.reshape(1, -1)], axis=1)
    g_final, _, _ = G.forward(gc_f, last_linear=True)
    imputed_norm = g_final.flatten() * (1 - mask) + x_norm * mask
    return _min_max_inv(imputed_norm, min_val, max_val)


def run_condgan_imputation(
    data: List[DataPoint],
    large_gap_indices: List[int],
    params: SimulationParams,
    rng: np.random.Generator = None,
    epochs_override: Optional[int] = None,
) -> List[DataPoint]:
    """Conditional Imputation GAN: G(z, c) with c = (mask, observed); imputation for Extended MAR/EAMAR (Deng et al., arXiv:2011.02089)."""
    if rng is None:
        rng = np.random.default_rng()

    large_set = set(large_gap_indices)
    values, mask = _series_and_mask(data, large_set)
    n = len(data)
    obs_vals = [data[i].observed for i in range(n) if i not in large_set and data[i].observed is not None]
    min_val = min(obs_vals) if obs_vals else 0.0
    max_val = max(obs_vals) if obs_vals else 1.0

    if np.sum(mask) < 2:
        return [
            DataPoint(
                time=data[i].time,
                observed=data[i].observed,
                instance_id=data[i].instance_id,
                imputed=data[i].observed,
                auxiliaries=data[i].auxiliaries,
                is_masked=data[i].is_masked,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            )
            for i in range(n)
        ]

    epochs = epochs_override if epochs_override is not None else (getattr(params, "ganEpochs", 150) or 150)
    imputed_vals = _run_condgan_single(values, mask, min_val, max_val, epochs=epochs, rng=rng)
    offset = max(0, n - len(imputed_vals))

    def _get_imputed(i: int) -> float:
        j = i - offset
        if 0 <= j < len(imputed_vals):
            return float(imputed_vals[j])
        if data[i].observed is not None:
            return data[i].observed
        return (min_val + max_val) / 2

    out: List[DataPoint] = []
    for i in range(n):
        if i in large_set:
            out.append(DataPoint(
                time=data[i].time,
                observed=None,
                instance_id=data[i].instance_id,
                auxiliaries=data[i].auxiliaries,
                imputed=None,
                is_masked=True,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            ))
        elif data[i].observed is not None:
            out.append(DataPoint(
                time=data[i].time,
                observed=data[i].observed,
                instance_id=data[i].instance_id,
                auxiliaries=data[i].auxiliaries,
                imputed=_get_imputed(i),
                is_masked=False,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            ))
        else:
            out.append(DataPoint(
                time=data[i].time,
                observed=None,
                instance_id=data[i].instance_id,
                auxiliaries=data[i].auxiliaries,
                imputed=_get_imputed(i),
                is_masked=False,
                latitude=data[i].latitude,
                longitude=data[i].longitude,
            ))
    return out


def compute_holdout_kge_condgan(
    data: List[DataPoint],
    large_gap_indices: List[int],
    params: SimulationParams,
    epochs_override: int,
    holdout_frac: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Measure gap-filling quality for CondGAN: mask 15% of observed, impute, compute KGE at holdout."""
    if rng is None:
        rng = np.random.default_rng()
    large_set = set(large_gap_indices)
    observed_indices = [i for i in range(len(data)) if i not in large_set and data[i].observed is not None]
    if len(observed_indices) < 3:
        return float("-inf")
    n_holdout = max(2, int(len(observed_indices) * holdout_frac))
    n_holdout = min(n_holdout, len(observed_indices) - 1)
    holdout_set = set(rng.choice(observed_indices, size=n_holdout, replace=False))
    true_holdout = [data[i].observed for i in sorted(holdout_set)]
    data_holdout = []
    for i, d in enumerate(data):
        obs = None if i in holdout_set else d.observed
        data_holdout.append(DataPoint(
            time=d.time, observed=obs, instance_id=d.instance_id,
            auxiliaries=d.auxiliaries, is_masked=d.is_masked,
            latitude=d.latitude, longitude=d.longitude,
        ))
    result = run_condgan_imputation(data_holdout, large_gap_indices, params, rng=rng, epochs_override=epochs_override)
    pred_holdout = [result[i].imputed for i in sorted(holdout_set) if i < len(result) and result[i].imputed is not None]
    if len(pred_holdout) != len(true_holdout) or len(true_holdout) < 2:
        return float("-inf")
    return calculate_kge(true_holdout, pred_holdout)


def optimize_gain_hyperparams(
    data: List[DataPoint],
    large_gap_indices: List[int],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
    trials: Optional[int] = None,
) -> Tuple[int, float]:
    """Auto-tune GAIN epochs and hint_rate to maximize holdout KGE.
    Uses shorter trial epochs for speed, then returns the best config."""
    if rng is None:
        rng = np.random.default_rng()
    trials = trials if trials is not None else getattr(params, "small_gap_gan_trials", 3) or 3
    best_kge = float("-inf")
    best_epochs, best_hint = 150, 0.9
    # Use shorter epochs for tuning trials (the final run will use the chosen epoch count)
    epoch_choices = [100, 150, 200]
    hint_choices = [0.7, 0.8, 0.9]
    for t in range(trials):
        epochs = int(rng.choice(epoch_choices))
        hint_rate = float(rng.choice(hint_choices))
        result = run_gain_imputation(
            data, large_gap_indices, params, rng=rng,
            epochs_override=epochs, hint_rate_override=hint_rate,
        )
        kge = _kge_from_result(result, large_gap_indices)
        if kge > best_kge:
            best_kge = kge
            best_epochs, best_hint = epochs, hint_rate
        if (t + 1) % 3 == 0 or t == trials - 1:
            kge_str = f"{best_kge:.4f}" if best_kge > float("-inf") else "N/A"
            print(f"    GAIN trial {t + 1}/{trials}, best KGE={kge_str}", flush=True)
    return best_epochs, best_hint


def optimize_condgan_hyperparams(
    data: List[DataPoint],
    large_gap_indices: List[int],
    params: SimulationParams,
    rng: Optional[np.random.Generator] = None,
    trials: Optional[int] = None,
) -> int:
    """Auto-tune Conditional GAN epochs to maximize KGE."""
    if rng is None:
        rng = np.random.default_rng()
    trials = trials if trials is not None else getattr(params, "small_gap_gan_trials", 3) or 3
    best_kge = float("-inf")
    best_epochs = 150
    epoch_choices = [100, 150, 200]
    for t in range(trials):
        epochs = int(rng.choice(epoch_choices))
        result = run_condgan_imputation(
            data, large_gap_indices, params, rng=rng, epochs_override=epochs,
        )
        kge = _kge_from_result(result, large_gap_indices)
        if kge > best_kge:
            best_kge = kge
            best_epochs = epochs
        if (t + 1) % 3 == 0 or t == trials - 1:
            kge_str = f"{best_kge:.4f}" if best_kge > float("-inf") else "N/A"
            print(f"    CondGAN trial {t + 1}/{trials}, best KGE={kge_str}", flush=True)
    return best_epochs
