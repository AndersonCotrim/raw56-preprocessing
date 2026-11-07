from pathlib import Path
import numpy as np

import pandas as pd

DATASET_FOLDER = Path("G:\\Final3")

df = pd.read_csv(DATASET_FOLDER / "scores_per_scene.csv")

models = [
    "pretrained_burstsr",
    "pretrained_raw56_14frames",
    "pretrained_raw56_allframes",
]

metrics = ["psnr", "ssim", "lpips"]

# Mean and standard deviation for each model and frame count
table = df.groupby(["nf", "chk"])[metrics].agg(["mean", "std"])

# Format mean ± STD
table = pd.DataFrame({
    metric.upper(): table[(metric, "mean")].map(
        lambda x: f"{x:.2f}" if metric == "psnr" else f"{x:.4f}"
    ) + " ± " + table[(metric, "std")].map(
        lambda x: f"{x:.2f}" if metric == "psnr" else f"{x:.4f}"
    )
    for metric in metrics
})

# Organize models as columns
table = table.unstack("chk")
table = table.reindex(columns=models, level=1)
table = table.swaplevel(0, 1, axis=1)

print(table.to_string())


df = df[df["nf"] == 56]

model14 = df[df["chk"] == "pretrained_raw56_14frames"]
model56 = df[df["chk"] == "pretrained_raw56_allframes"]

# Match the same test scenes
paired = model14.merge(model56, on="scene", suffixes=("_14", "_56"))

assert len(paired) == 200, "Expected 200 paired test scenes"

for metric in ["psnr", "ssim", "lpips"]:
    if metric == "lpips":
        improved = paired[f"{metric}_56"] < paired[f"{metric}_14"]
    else:
        improved = paired[f"{metric}_56"] > paired[f"{metric}_14"]

    print(
        f"{metric.upper()}: {improved.sum()}/200 scenes "
        f"({improved.mean() * 100:.1f}%) improved"
    )

    print(f"  {metric.upper()} mean improvement: {paired[f'{metric}_56'].mean() - paired[f'{metric}_14'].mean():.4f}")


rng = np.random.default_rng(42)

for metric in ["psnr", "ssim", "lpips"]:
    delta = (paired[f"{metric}_14"] - paired[f"{metric}_56"]
             if metric == "lpips"
             else paired[f"{metric}_56"] - paired[f"{metric}_14"])

    samples = rng.choice(delta.to_numpy(), (10000, len(delta)), replace=True)
    ci = np.percentile(samples.mean(axis=1), [2.5, 97.5])

    print(f"{metric.upper()}: gain={delta.mean():+.6f}, 95% CI={ci}")