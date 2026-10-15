# %%
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import laion_fmri.splits as splits
import seaborn as sns
from matplotlib import pyplot as plt

from lib.datasets import (
    compute_shared_stimuli,
    filter_by_stimulus,
    laion,
    split_by_repetition,
)
from lib.datasets.laion import get_split_stimuli
from lib.spectra import (
    compute_cross_individual_spectra,
    compute_within_individual_spectra,
    plot_spectra,
)
from lib.utilities import JOURNAL_MATPLOTLIBRC, mathtext_exponent_label

# ------- settings
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
REFERENCE_SUBJECT = 0
FIGURES_HOME = Path.cwd().parent / "figures-laion"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

METHOD = "ood"          # "tau", "cluster", "random" or "ood"
N_PERMUTATIONS = 5000     # 10 for testing, 5000 for the final run

METHOD_LABELS = {
    "tau": "Method 1: independent within-distribution (tau, 80/20)",
    "cluster": "Method 2: out-of-distribution clusters (5-fold, mean)",
    "random": "Random baseline (5-fold, mean)",
    "ood": "Method 3: out-of-distribution images (train regular, test OOD)",
}
SPLIT_NAMES = {
    "tau": ["tau"],
    "cluster": [f"cluster_k5_{k}" for k in range(5)],
    "random": [f"random_{k}" for k in range(5)],
    "ood": ["ood"],
}[METHOD]

# OOD categories to loop over (None = all 9 together)
OOD_TYPES = [None, *splits.list_ood_types(), ["illusion-classic", "illusion-natural"]]

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

# ------- load datasets (once)
datasets = {
    i: laion.load_dataset(subject=s, roi="general", include_ood=(METHOD == "ood"))
    for i, s in enumerate(SUBJECTS)
}

datasets_within = {
    subject: split_by_repetition(
        filter_by_stimulus(
            dataset,
            stimuli=compute_shared_stimuli([dataset], n_repetitions=2),
        ),
        n_repetitions=2,
    )
    for subject, dataset in datasets.items()
}

shared_stimuli = compute_shared_stimuli(datasets.values(), n_repetitions=2)

datasets_cross = {
    subject: split_by_repetition(
        filter_by_stimulus(dataset, stimuli=shared_stimuli),
        n_repetitions=2,
    )
    for subject, dataset in datasets.items()
}


def make_figure(ood_type=None):
    # --- splits (ood_types only matters for METHOD == "ood")
    splits_within = {
        i: [get_split_stimuli(subject=s, pool=s, name=n, ood_types=ood_type) for n in SPLIT_NAMES]
        for i, s in enumerate(SUBJECTS)
    }
    splits_cross = [
        get_split_stimuli(
            subject=SUBJECTS[REFERENCE_SUBJECT], pool="shared", name=n, ood_types=ood_type
        )
        for n in SPLIT_NAMES
    ]

    # --- label for title + file name
    if ood_type is None:
        label = "all"
    elif isinstance(ood_type, list):
        label = "+".join(ood_type)
    else:
        label = ood_type
    n_test = len(splits_cross[0][1])

    if METHOD == "ood":
        title = f"{METHOD_LABELS[METHOD]}\n{label} (n = {n_test} test images)"
        suffix = f"_{label}"
    else:
        title = METHOD_LABELS[METHOD]
        suffix = ""

    # compute spectra
    spectra_within = compute_within_individual_spectra(
        datasets_within,
        splits=splits_within,
        n_permutations=N_PERMUTATIONS,
    )
    spectra_cross = compute_cross_individual_spectra(
        datasets_cross,
        reference_individual=REFERENCE_SUBJECT,
        splits=splits_cross,
        n_permutations=N_PERMUTATIONS,
        randomized=True,
    )

    # plot
    fig, axes = plt.subplots(figsize=(5.5, 3), ncols=2, sharex=True, sharey=True)

    kwargs_legend = {
        "loc": "lower left",
        "title": "subject",
        "ncols": 2,
        "columnspacing": 0.5,
        "handletextpad": 0.25,
        "reverse": True,
    }

    ax = axes[0]
    plot_spectra(
        ax=ax,
        spectra=spectra_within,
        hue="individual",
        palette="crest",
        hue_order=list(reversed(range(len(SUBJECTS)))),
        hue_labels=[SUBJECTS[subject] for subject in reversed(range(len(SUBJECTS)))],
        marker="s",
        hide_insignificant=True,
        null_quantile=0.999,
    )
    ax.set_title("within-subject", pad=10)
    ax.set_ylabel("covariance")
    ax.set_xlabel("rank")
    ax.legend(**kwargs_legend)

    ax = axes[1]
    plot_spectra(
        ax=ax,
        spectra=spectra_cross,
        hue="individual",
        palette="flare",
        hue_reference=REFERENCE_SUBJECT,
        hue_order=list(reversed(range(len(SUBJECTS)))),
        hue_labels=[
            f"{SUBJECTS[subject]}*" if subject == REFERENCE_SUBJECT else SUBJECTS[subject]
            for subject in reversed(range(len(SUBJECTS)))
        ],
        marker=None,
        hide_insignificant=True,
        null_quantile=0.999,
    )
    ax.set_title(f"between-subject,\nrelative to subject {SUBJECTS[REFERENCE_SUBJECT]}")
    ax.set_ylabel("cross-covariance")
    ax.set_xlabel("rank")
    if METHOD != "ood":
        ax.axvline(len(shared_stimuli), ls="--", c="gray", lw=0.5, ymax=0.45)
        ax.text(
            s="number of\nshared images",
            x=len(shared_stimuli),
            y=2e-9,
            fontsize="xx-small",
            ha="center",
            va="bottom",
        )
    ax.legend(**kwargs_legend)
    ax.yaxis.set_tick_params(labelbottom=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1, right=1e4)
    ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4])
    ax.set_ylim(top=1e-1, bottom=1e-9)
    ytick_exponents = list(range(-9, 0))
    ax.set_yticks(
        [10**exponent for exponent in ytick_exponents],
        labels=[
            mathtext_exponent_label(exponent) if exponent % 2 == 1 else ""
            for exponent in ytick_exponents
        ],
    )

    fig.suptitle(title)
    name = f"general_laion_{METHOD}{suffix}_perm_{N_PERMUTATIONS}"
    fig.savefig(FIGURES_HOME / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_HOME / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if METHOD == "ood":
    for t in OOD_TYPES:
        print(f"--- OOD type: {t}", flush=True)
        make_figure(t)
else:
    make_figure()