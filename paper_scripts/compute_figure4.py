from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    DATASET_FOLDER = Path("G:\\Final3")

    H_norm = [np.load(p)[1:] for p in (DATASET_FOLDER / 'data').glob("**/h_norm.npy")]
    H = np.load("H.npy")
    H_norm = np.asarray(H_norm).reshape(-1, 3, 3)

    Path("figs").mkdir(exist_ok=True)
    sns.set_style("white")


    tx = H[:, 0, 2]
    ty = H[:, 1, 2]
    tx_norm = H_norm[:, 0, 2]
    ty_norm = H_norm[:, 1, 2]

    plot_distribution(tx, ty, lims=20, name="figs/pixel_dist.png")
    plot_distribution(tx_norm, ty_norm, lims=5, name="figs/pixel_dist_norm.png")

    theta = np.degrees(
        np.arctan2(
            H[:, 1, 0] - H[:, 0, 1],
            H[:, 0, 0] + H[:, 1, 1]
        )
    )

    theta = theta[np.isfinite(theta)]

    print(f"Total transformations: {len(theta)}")
    print(f"Mean absolute rotation: {np.mean(np.abs(theta)):.4f}°")
    print(f"Median absolute rotation: {np.median(np.abs(theta)):.4f}°")
    print(f"Standard deviation: {np.std(theta):.4f}°")
    print(f"95th percentile (absolute): {np.percentile(np.abs(theta), 95):.4f}°")
    print(f"Maximum absolute rotation: {np.max(np.abs(theta)):.4f}°")

    print(f"|θ| < 0.1°: {np.mean(np.abs(theta) < 0.1) * 100:.2f}%")
    print(f"|θ| < 0.5°: {np.mean(np.abs(theta) < 0.5) * 100:.2f}%")
    print(f"|θ| < 1.0°: {np.mean(np.abs(theta) < 1.0) * 100:.2f}%")

    plot_rotation_distribution(theta)


def plot_rotation_distribution(theta, name="figs/rotation_dist.png"):
    sns.set_style("white")

    theta = np.asarray(theta)
    theta = theta[np.isfinite(theta)]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    sns.histplot(
        theta,
        bins=60,
        stat="density",
        kde=True,
        color="#9DCFC6",
        edgecolor=None,
        ax=ax
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(r"Rotation angle $\theta$ (degrees)")
    ax.set_ylabel("Density")

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_distribution(tx, ty, lims=20, name="jointplot_tx_ty.png"):
    data = pd.DataFrame({
        'tx': tx,
        'ty': ty,
    })

    g = sns.jointplot(data=data, x="tx", y="ty", height=3.5, ratio=4, kind="kde", fill=True, levels=12, thresh=0.05,
                      color="#9DCFC6")

    g.ax_joint.set_xlim(-lims, lims)
    g.ax_joint.set_ylim(-lims, lims)

    g.ax_marg_x.set_xlim(-lims, lims)
    g.ax_marg_y.set_ylim(-lims, lims)

    g.ax_marg_x.cla()
    g.ax_marg_y.cla()

    sns.histplot(
        data=data, x="tx", bins=50, ax=g.ax_marg_x,
        color="#9DCFC6", legend=False, binrange=(-lims, lims),
        element="bars", linewidth=0, edgecolor=None
    )
    sns.histplot(
        data=data, y="ty", bins=50, ax=g.ax_marg_y,
        color="#9DCFC6", legend=False, binrange=(-lims, lims),
        element="bars", linewidth=0, edgecolor=None
    )

    g.ax_joint.set_xlabel("$t_x$ (pixels)")
    g.ax_joint.set_ylabel("$t_y$ (pixels)")

    g.ax_joint.grid(False)
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)

    for ax in (g.ax_marg_x, g.ax_marg_y):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    g.fig.savefig(name, dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()