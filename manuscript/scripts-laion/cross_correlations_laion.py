import matplotlib
matplotlib.use("Agg")

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from bonner.plotting import save_figure
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from lib.datasets import (
    compute_shared_stimuli,
    filter_by_stimulus,
    laion,
    split_by_repetition,
)
from lib.datasets.laion_mni import convert_datasets_to_mni_space
from lib.spectra import compute_spectra_with_n_fold_cross_validation
from lib.utilities import JOURNAL_MATPLOTLIBRC

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
FIGURES_HOME = Path.cwd().parent / "figures-laion"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

REFERENCE_SUBJECT = 0
RANK_RANGES = ((0, 10), (10, 50), (50, 100))
N_FOLDS = 8

# %% [markdown]
# ## load datasets and convert to MNI305

# %%
datasets = {i: laion.load_dataset(subject=s, roi="general") for i, s in enumerate(SUBJECTS)}

shared_stimuli = compute_shared_stimuli(datasets.values(), n_repetitions=2)
datasets_cross = {
    subject: split_by_repetition(
        filter_by_stimulus(dataset, stimuli=shared_stimuli),
        n_repetitions=2,
    )
    for subject, dataset in datasets.items()
}

datasets_mni = convert_datasets_to_mni_space(
    datasets_cross,
    subject_ids=dict(enumerate(SUBJECTS)),
    order=1,
    resolution=1.8,
)

for subject, datasets_ in datasets_mni.items():
    print(
        SUBJECTS[subject],
        "native voxels:", datasets_cross[subject][0].sizes["neuroid"],
        "shared MNI voxels:", datasets_[0].sizes["neuroid"],
        "stimuli:", datasets_[0].sizes["presentation"],
    )



# %%
def compute_pair_spectra(
    dataset_1: dict[int, xr.DataArray],
    dataset_2: dict[int, xr.DataArray],
) -> xr.Dataset:
    spectra = []
    for comparison, (r1, r2) in enumerate(((0, 1), (1, 0))):
        spectra.append(
            compute_spectra_with_n_fold_cross_validation(
                x_train=dataset_1[r1],
                y_train=dataset_2[r2],
                x_test=dataset_1[r1],
                y_test=dataset_2[r2],
                n_folds=N_FOLDS,
                randomized=True,
            ).expand_dims({"comparison": [comparison], "alignment": ["functional"]}),
        )
        spectra.append(
            compute_spectra_with_n_fold_cross_validation(
                x_train=dataset_1[r1],
                y_train=dataset_2[r2],
                x_test=dataset_2[r2],
                y_test=dataset_1[r1],
                n_folds=N_FOLDS,
                randomized=True,
            ).expand_dims({"comparison": [comparison], "alignment": ["anatomical"]}),
        )
    return xr.merge(spectra)


def sum_rank_ranges(
    spectrum: xr.Dataset,
    rank_ranges: Sequence[tuple[int, int]],
) -> xr.Dataset:
    return xr.concat(
        [
            spectrum.isel(component=slice(start, stop))
            .sum("component")
            .expand_dims({"range": [i_range]})
            .assign_coords({"start": ("range", [start + 1]), "stop": ("range", [stop])})
            for i_range, (start, stop) in enumerate(rank_ranges)
        ],
        dim="range",
    )


spectra = {}
for (s1, d1), (s2, d2) in tqdm(
    list(itertools.combinations_with_replacement(datasets_mni.items(), r=2)),
    desc="pair of subjects",
):
    spectra[frozenset([s1, s2])] = sum_rank_ranges(
        compute_pair_spectra(d1, d2),
        RANK_RANGES,
    ).mean("comparison")


# %% [markdown]
# ## cross-correlations r_k = Σ(X,Y) / sqrt(Σ(X1,X2) Σ(Y1,Y2))

# %%
def compute_cross_correlations(reference: int) -> pd.DataFrame:
    return pd.concat(
        [
            (
                spectra[frozenset([reference, subject])]
                / np.sqrt(
                    spectra[frozenset([subject, subject])]
                    * spectra[frozenset([reference, reference])],
                )
            )
            .expand_dims({"subject": [subject]})
            .rename({"covariance": "correlation"})
            .to_dataframe()
            .reset_index()
            .dropna()
            for subject in datasets_mni
            if subject != reference
        ],
    )


correlations = {reference: compute_cross_correlations(reference) for reference in datasets_mni}
pd.concat(
    [df.assign(reference=SUBJECTS[r]) for r, df in correlations.items()],
).to_csv(FIGURES_HOME / "cross_correlations_laion.csv", index=False)


# %% [markdown]
# ## plot

# %%
def plot_cross_correlations(reference: int) -> plt.Figure:
    palette = sns.color_palette("flare_r", n_colors=len(SUBJECTS))
    others = [s for s in range(len(SUBJECTS)) if s != reference]
    df = correlations[reference]

    fig, axes = plt.subplots(
        figsize=(5.5, 3),
        ncols=len(RANK_RANGES) + 1,
        sharey=True,
        squeeze=False,
        width_ratios=[1] * len(RANK_RANGES) + [0.5],
    )

    for i_range, ((start, stop), ax) in enumerate(zip(RANK_RANGES, axes.flat, strict=False)):
        for alignment in ("functional", "anatomical"):
            for i_tick, subject in enumerate(others, start=1):
                values = df.loc[
                    (df["range"] == i_range)
                    & (df["alignment"] == alignment)
                    & (df["subject"] == subject),
                    "correlation",
                ]
                mean, std = values.mean(), values.std()
                significant = (mean - 3 * std / np.sqrt(len(values) - 1)) > 0
                ax.errorbar(
                    i_tick * (1 / 1.025 if alignment == "functional" else 1.025),
                    mean,
                    std,
                    ls="None",
                    marker="o" if alignment == "functional" else "^",
                    c=palette[subject],
                    mew=0 if significant else 1,
                    mfc=palette[subject] if significant else "none",
                    ms=6 if alignment == "anatomical" else 5,
                    alpha=1 if alignment == "functional" else 0.5,
                )

        ax.set_xticks(
            1 + np.arange(len(others)),
            [SUBJECTS[s].removeprefix("sub-") for s in others],
        )
        ax.set_ylim(bottom=-0.2, top=1.05)
        ax.axhline(1, ls="--", c="gray", lw=0.5)
        ax.axhline(0, ls="--", c="gray", lw=0.5)
        ax.set_title(f"rank {start + 1} to {stop}")

    axes.flat[0].set_ylabel("cross-correlation")

    ax = axes.flat[-1]
    ax.errorbar(0, 0, 0, ls="None", marker="o", mew=0, c="gray", ms=5, label="functional")
    ax.errorbar(0, 0, 0, ls="None", marker="^", mew=0, c="gray", ms=6, label="anatomical")
    ax.legend(title="alignment", loc="center left", borderaxespad=0)
    ax.set_xlim(left=1, right=2)
    ax.axis("off")

    fig.supxlabel("subject", x=0.45, y=0.06)
    fig.suptitle(
        f"between-subject similarity, relative to {SUBJECTS[reference]}",
        x=0.45,
    )
    return fig


for reference in datasets_mni:
    fig = plot_cross_correlations(reference)
    save_figure(fig, filepath=FIGURES_HOME / f"cross_correlations_{SUBJECTS[reference]}.pdf")
    if reference == REFERENCE_SUBJECT:
        save_figure(fig, filepath=FIGURES_HOME / "cross_correlations_laion.pdf")
    plt.close(fig)
