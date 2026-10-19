# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # spectral normalization and number of stimuli (Figs S1, S2)

# %%
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from bonner.plotting import save_figure
from matplotlib import pyplot as plt

from lib.datasets import (
    compute_shared_stimuli,
    filter_by_stimulus,
    laion,
    split_by_repetition,
)
from lib.spectra import compute_within_individual_spectra
from lib.utilities import JOURNAL_MATPLOTLIBRC

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
SUBJECT = "sub-01"
VOXEL_COUNTS = (100, 316, 1000, 3162, 10000)
N_STIMULUS_COUNTS = 5
N_FOLDS = 8
SEED = 0
FIGURES_HOME = Path.cwd().parent / "figures-laion"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

# %% [markdown]
# ## load data

# %%
dataset = laion.load_dataset(subject=SUBJECT, roi="general")
stimuli = np.array(sorted(compute_shared_stimuli([dataset], n_repetitions=2)))
full = split_by_repetition(filter_by_stimulus(dataset, stimuli=stimuli), n_repetitions=2)
n_voxels_total = full[0].sizes["neuroid"]
n_stimuli_total = full[0].sizes["presentation"]

n_shared = len(
    compute_shared_stimuli(
        [laion.load_dataset(subject=s, roi="general") for s in SUBJECTS],
        n_repetitions=2,
    ),
)

voxel_counts = [n for n in VOXEL_COUNTS if n < n_voxels_total] + [n_voxels_total]
stimulus_counts = sorted(
    {int(n) for n in np.geomspace(n_shared, n_stimuli_total, N_STIMULUS_COUNTS).round()},
)
print(SUBJECT, "voxels:", n_voxels_total, "stimuli:", n_stimuli_total, "shared across subjects:", n_shared)
print("voxel counts:", voxel_counts)
print("stimulus counts:", stimulus_counts)


# %% [markdown]
# ## subsampling (each subset gets its own name so the spectra cache does not reuse results)

# %%
def tag(split: dict[int, xr.DataArray], label: str) -> dict[int, xr.DataArray]:
    return {r: data.rename(f"{data.name}.{label}") for r, data in split.items()}


def subsample_voxels(n_voxels: int) -> dict[int, xr.DataArray]:
    rng = np.random.default_rng(SEED)
    voxels = np.sort(rng.choice(n_voxels_total, size=n_voxels, replace=False))
    return tag(
        {r: data.isel(neuroid=voxels) for r, data in full.items()},
        f"voxels={n_voxels}.seed={SEED}",
    )


def subsample_stimuli(n_stimuli: int) -> dict[int, xr.DataArray]:
    rng = np.random.default_rng(SEED)
    subset = rng.choice(stimuli, size=n_stimuli, replace=False)
    split = split_by_repetition(filter_by_stimulus(dataset, stimuli=subset), n_repetitions=2)
    return tag(split, f"stimuli={n_stimuli}.seed={SEED}")


# %% [markdown]
# ## spectra

# %%
spectra_voxels = {
    normalize: compute_within_individual_spectra(
        {n: subsample_voxels(n) for n in voxel_counts},
        n_folds=N_FOLDS,
        randomized=True,
        normalize=normalize,
    )
    for normalize in (False, True)
}

spectra_stimuli = compute_within_individual_spectra(
    {n: subsample_stimuli(n) for n in stimulus_counts},
    n_folds=N_FOLDS,
    randomized=True,
)


def summarize(spectra: xr.Dataset) -> pd.DataFrame:
    covariance = spectra["covariance"]
    mean = covariance.mean("fold")
    std = covariance.std("fold")
    sem = std / np.sqrt(covariance.sizes["fold"] - 1)
    return (
        xr.Dataset({"mean": mean, "std": std, "significant": (mean - 3 * sem) > 0})
        .to_dataframe()
        .reset_index()
        .dropna(subset=["mean"])
        .rename(columns={"individual": "count"})
    )


summaries = {
    "voxels_before": summarize(spectra_voxels[False]),
    "voxels_after": summarize(spectra_voxels[True]),
    "stimuli": summarize(spectra_stimuli),
}
pd.concat([df.assign(condition=name) for name, df in summaries.items()]).to_csv(
    FIGURES_HOME / f"spectra_subsampling_{SUBJECT}.csv",
    index=False,
)


# %% [markdown]
# ## plot

# %%
def plot_counts(ax: plt.Axes, df: pd.DataFrame, counts: list[int], palette: list) -> None:
    for i, (n, c) in enumerate(zip(counts, palette, strict=True)):
        offset = 1.06 ** (i - (len(counts) - 1) / 2)
        df_ = df[(df["count"] == n) & (df["mean"] > 0)]
        for significant, df__ in df_.groupby("significant"):
            ax.errorbar(
                df__["rank"] * offset,
                df__["mean"],
                df__["std"],
                ls="None",
                marker="s",
                ms=4,
                c=c,
                mew=0 if significant else 1,
                mfc=c if significant else "none",
            )
        ax.plot([], [], ls="None", marker="s", ms=4, c=c, label=f"{n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4])


palette_voxels = sns.color_palette("viridis_r", n_colors=len(voxel_counts))
fig, axes = plt.subplots(ncols=2, figsize=(6, 3), sharex=True)
for ax, key, title in zip(
    axes,
    ("voxels_before", "voxels_after"),
    ("before normalization", "after normalization"),
    strict=True,
):
    plot_counts(ax, summaries[key], voxel_counts, palette_voxels)
    ax.set_title(title)
axes[0].set_ylabel("covariance")
axes[1].legend(title="# voxels", loc="lower left", fontsize="small")
fig.supxlabel("rank")
fig.suptitle(f"within-subject, {SUBJECT}")
fig.tight_layout()
save_figure(fig, filepath=FIGURES_HOME / f"spectral_normalization_{SUBJECT}.pdf")
plt.close(fig)

palette_stimuli = sns.color_palette("inferno", n_colors=len(stimulus_counts))
fig, ax = plt.subplots(figsize=(3.5, 3.5))
plot_counts(ax, summaries["stimuli"], stimulus_counts, palette_stimuli)
ax.set_xlabel("rank")
ax.set_ylabel("covariance")
ax.set_title(f"within-subject, {SUBJECT}")
ax.legend(title="number of stimuli", loc="upper right", fontsize="small")
fig.tight_layout()
save_figure(fig, filepath=FIGURES_HOME / f"number_of_stimuli_{SUBJECT}.pdf")
plt.close(fig)
