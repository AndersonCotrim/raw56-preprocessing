"""Paired SIFT vs. ECC statistics for the paper (scene-level data)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_FOLDER = Path("G:\\Final3")

CSV_FILE = DATASET_FOLDER  / 'scene_metadata.csv'
N_BOOT = 10_000
rng = np.random.default_rng(42)

df = pd.read_csv(CSV_FILE)

for metric, sift_col, ecc_col, lower_is_better in [
    ("Geometric RMSE (LR pixels)", "sift_rmse_px", "ecc_rmse_px", True),
    ("Gradient correlation", "grad_corr_sift", "grad_corr_ecc", False),
]:
    paired = df[[sift_col, ecc_col]].dropna()
    sift = paired[sift_col].to_numpy()
    ecc = paired[ecc_col].to_numpy()
    gain = sift - ecc if lower_is_better else ecc - sift
    bootstrap_means = rng.choice(gain, size=(N_BOOT, len(gain)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    print(f"\n{metric} (N={len(paired)})")
    print(f"  SIFT:         {sift.mean():.6f} ± {sift.std(ddof=1):.6f} (median {np.median(sift):.6f})")
    print(f"  ECC:          {ecc.mean():.6f} ± {ecc.std(ddof=1):.6f} (median {np.median(ecc):.6f})")
    print(f"  ECC gain:     {gain.mean():+.6f} (positive = better)")
    print(f"  ECC improved: {(gain > 0).mean() * 100:.1f}% of scenes")
    print(f"  95% CI:       [{ci_low:+.6f}, {ci_high:+.6f}] (paired bootstrap, {N_BOOT:,} resamples)")
    print(f"  SIFT P5:      {np.percentile(sift, 5):.6f}")
    print(f"  ECC P5:       {np.percentile(ecc, 5):.6f}")
    print(f"  SIFT P95:     {np.percentile(sift, 95):.6f}")
    print(f"  ECC P95:      {np.percentile(ecc, 95):.6f}")
    print(f"  Gain STD:     {gain.std(ddof=1):.6f}")
    print(f"  Gain median:  {np.median(gain):+.6f}")
