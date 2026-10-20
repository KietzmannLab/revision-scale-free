import matplotlib
matplotlib.use("Agg")

import itertools
import os
from collections.abc import Sequence
from pathlib import Path

import laion_fmri.splits as splits
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
from lib.datasets import laion_mni, laion_mni_template
from lib.datasets.laion import get_split_stimuli
from lib.spectra import compute_spectrum
from lib.utilities import JOURNAL_MATPLOTLIBRC

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
FIGURES_HOME = Path.cwd().parent / "figures-laion" / "ood"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

RANK_RANGES = ((0, 10), (10, 50), (50, 100))
N_BOOTSTRAPS = int(os.environ.get("N_BOOTSTRAPS", "1000"))
BASELINE_SPLIT = "random_0"
METHOD = os.environ.get("MNI_METHOD", "linear")
SUFFIX = "" if METHOD == "linear" else f"_{METHOD}"
OOD_TYPES = (
    [None]
    if os.environ.get("OOD_TYPES", "all") == "all"
    else [None, *splits.list_ood_types(), ["illusion-classic", "illusion-natural"]]
)

datasets = {
    i: laion.load_dataset(subject=s, roi="general", include_ood=True)
    for i, s in enumerate(SUBJECTS)
}
shared_stimuli = compute_shared_stimuli(datasets.values(), n_repetitions=2)
datasets_cross = {
    subject: split_by_repetition(
        filter_by_stimulus(dataset, stimuli=shared_stimuli),
        n_repetitions=2,
    )
    for subject, dataset in datasets.items()
}

if METHOD == "to_template":
    datasets_mni = laion_mni_template.convert_datasets_to_mni_space(
        datasets_cross,
        subject_ids=dict(enumerate(SUBJECTS)),
        resolution=1.8,
    )
else:
    datasets_mni = laion_mni.convert_datasets_to_mni_space(
        datasets_cross,
        subject_ids=dict(enumerate(SUBJECTS)),
        order=1,
        resolution=1.8,
    )

stimuli = datasets_mni[0][0]["stimulus"].values
for datasets_ in datasets_mni.values():
    for repetition in (0, 1):
        assert np.array_equal(datasets_[repetition]["stimulus"].values, stimuli)

is_ood = datasets_mni[0][0]["dataset"].values == "OOD"
print(
    "shared stimuli:", len(stimuli),
    "regular:", int((~is_ood).sum()),
    "OOD:", int(is_ood.sum()),
    "shared MNI voxels:", datasets_mni[0][0].sizes["neuroid"],
)


def get_masks(
    name: str,
    *,
    ood_types=None,
    n_test: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    train, test = get_split_stimuli(
        subject=SUBJECTS[0],
        pool="shared",
        name=name,
        ood_types=ood_types,
    )
    test = np.intersect1d(test, stimuli)
    if n_test is not None and len(test) > n_test:
        test = np.random.default_rng(seed).choice(test, size=n_test, replace=False)
    filter_train = np.isin(stimuli, train)
    filter_test = np.isin(stimuli, test)
    assert not (filter_train & filter_test).any()
    return filter_train, filter_test


def sum_rank_ranges(
    spectrum: xr.Dataset,
    rank_ranges: Sequence[tuple[int, int]],
) -> xr.Dataset:
    return xr.concat(
        [
            spectrum.isel(component=slice(start, stop))
            .sum("component")
            .expand_dims({"range": [i_range]})
            for i_range, (start, stop) in enumerate(rank_ranges)
        ],
        dim="range",
    )


def compute_pair_spectra(
    dataset_1: dict[int, xr.DataArray],
    dataset_2: dict[int, xr.DataArray],
    *,
    filter_train: np.ndarray,
    filter_test: np.ndarray,
) -> xr.Dataset:
    spectra = []
    for comparison, (r1, r2) in enumerate(((0, 1), (1, 0))):
        tests = {
            "functional": (dataset_1[r1], dataset_2[r2]),
            "anatomical": (dataset_2[r2], dataset_1[r1]),
        }
        for alignment, (x_test, y_test) in tests.items():
            spectra.append(
                compute_spectrum(
                    x_train=dataset_1[r1],
                    y_train=dataset_2[r2],
                    x_test=x_test,
                    y_test=y_test,
                    filter_train=filter_train,
                    filter_test=filter_test,
                    randomized=True,
                    n_bootstraps=N_BOOTSTRAPS,
                ).expand_dims({"comparison": [comparison], "alignment": [alignment]}),
            )
    return sum_rank_ranges(xr.merge(spectra), RANK_RANGES).mean("comparison")


def compute_all_spectra(
    filter_train: np.ndarray,
    filter_test: np.ndarray,
) -> dict[frozenset[int], xr.Dataset]:
    return {
        frozenset([s1, s2]): compute_pair_spectra(
            d1,
            d2,
            filter_train=filter_train,
            filter_test=filter_test,
        )
        for (s1, d1), (s2, d2) in tqdm(
            list(itertools.combinations_with_replacement(datasets_mni.items(), r=2)),
            desc="pair of subjects",
            leave=False,
        )
    }


def compute_cross_correlations(
    spectra: dict[frozenset[int], xr.Dataset],
    reference: int,
) -> pd.DataFrame:
    rows = []
    for subject in range(len(SUBJECTS)):
        if subject == reference:
            continue
        ratio = spectra[frozenset([reference, subject])] / np.sqrt(
            spectra[frozenset([subject, subject])]
            * spectra[frozenset([reference, reference])],
        )
        point = ratio["covariance"]
        bootstrapped = ratio["covariance (bootstrapped)"]
        bootstrap_dim = next(d for d in bootstrapped.dims if d not in point.dims)
        for i_range, alignment in itertools.product(range(len(RANK_RANGES)), ("functional", "anatomical")):
            values = bootstrapped.sel(range=i_range, alignment=alignment).transpose(bootstrap_dim).values
            values = values[np.isfinite(values)]
            rows.append({
                "reference": reference,
                "subject": subject,
                "range": i_range,
                "alignment": alignment,
                "correlation": float(point.sel(range=i_range, alignment=alignment)),
                "std": values.std() if len(values) > 1 else np.nan,
                "n_valid_bootstraps": len(values),
            })
    return pd.DataFrame(rows)


def plot_cross_correlations(
    correlations: dict[str, pd.DataFrame],
    n_tests: dict[str, int],
    reference: int,
    label: str,
) -> plt.Figure:
    palette = sns.color_palette("flare_r", n_colors=len(SUBJECTS))
    others = [s for s in range(len(SUBJECTS)) if s != reference]

    fig, axes = plt.subplots(
        figsize=(5.5, 5.5),
        nrows=len(correlations),
        ncols=len(RANK_RANGES) + 1,
        sharey=True,
        squeeze=False,
        width_ratios=[1] * len(RANK_RANGES) + [0.5],
    )

    for i_row, (condition, df) in enumerate(correlations.items()):
        df = df[df["reference"] == reference]
        for i_range, (start, stop) in enumerate(RANK_RANGES):
            ax = axes[i_row, i_range]
            ax.set_title(f"rank {start + 1} to {stop}")
            ax.set_ylim(bottom=-0.2, top=1.05)
            ax.axhline(1, ls="--", c="gray", lw=0.5)
            ax.axhline(0, ls="--", c="gray", lw=0.5)
            ax.set_xticks(
                1 + np.arange(len(others)),
                [SUBJECTS[s].removeprefix("sub-") for s in others],
            )
            if stop > n_tests[condition]:
                ax.text(
                    0.5, 0.5, "rank exceeds\n# test images",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize="x-small", c="gray",
                )
                continue

            for alignment in ("functional", "anatomical"):
                for i_tick, subject in enumerate(others, start=1):
                    row = df[
                        (df["range"] == i_range)
                        & (df["alignment"] == alignment)
                        & (df["subject"] == subject)
                    ].iloc[0]
                    significant = (row["correlation"] - 3 * row["std"]) > 0
                    ax.errorbar(
                        i_tick * (1 / 1.025 if alignment == "functional" else 1.025),
                        row["correlation"],
                        row["std"],
                        ls="None",
                        marker="o" if alignment == "functional" else "^",
                        c=palette[subject],
                        mew=0 if significant else 1,
                        mfc=palette[subject] if significant else "none",
                        ms=6 if alignment == "anatomical" else 5,
                        alpha=1 if alignment == "functional" else 0.5,
                    )

        axes[i_row, 0].set_ylabel(f"{condition}\n(n = {n_tests[condition]} test images)\ncross-correlation")
        axes[i_row, -1].axis("off")

    ax = axes[0, -1]
    ax.errorbar(0, 0, 0, ls="None", marker="o", mew=0, c="gray", ms=5, label="functional")
    ax.errorbar(0, 0, 0, ls="None", marker="^", mew=0, c="gray", ms=6, label="anatomical")
    ax.legend(title="alignment", loc="center left", borderaxespad=0)
    ax.set_xlim(left=1, right=2)

    fig.supxlabel("subject", x=0.45)
    fig.suptitle(
        f"between-subject similarity, relative to {SUBJECTS[reference]}\nOOD test set: {label}",
        x=0.45,
    )
    fig.tight_layout()
    return fig


for ood_type in OOD_TYPES:
    label = "all" if ood_type is None else ("+".join(ood_type) if isinstance(ood_type, list) else ood_type)
    print(f"--- OOD type: {label}", flush=True)

    masks = {"OOD": get_masks("ood", ood_types=ood_type)}
    n_test_ood = int(masks["OOD"][1].sum())
    masks["in-distribution"] = get_masks(BASELINE_SPLIT, n_test=n_test_ood)
    masks = {"in-distribution": masks["in-distribution"], "OOD": masks["OOD"]}
    n_tests = {condition: int(filter_test.sum()) for condition, (_, filter_test) in masks.items()}
    for condition, (filter_train, filter_test) in masks.items():
        print(condition, "train:", int(filter_train.sum()), "test:", int(filter_test.sum()), flush=True)

    correlations = {}
    for condition, (filter_train, filter_test) in masks.items():
        spectra = compute_all_spectra(filter_train, filter_test)
        correlations[condition] = pd.concat(
            [compute_cross_correlations(spectra, reference) for reference in range(len(SUBJECTS))],
        )

    pd.concat(
        [df.assign(condition=condition, n_test=n_tests[condition]) for condition, df in correlations.items()],
    ).assign(ood_type=label).to_csv(
        FIGURES_HOME / f"cross_correlations_ood_{label}{SUFFIX}.csv",
        index=False,
    )

    for reference in range(len(SUBJECTS)):
        fig = plot_cross_correlations(correlations, n_tests, reference, label)
        save_figure(
            fig,
            filepath=FIGURES_HOME / f"cross_correlations_ood_{label}_{SUBJECTS[reference]}{SUFFIX}.pdf",
        )
        plt.close(fig)
