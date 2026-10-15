# %%
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# NEW: only load the two ROI sources that exist locally (before any NSD loading!)
import bonner.datasets.allen2021_natural_scenes._data as nsd_data
nsd_data.ROI_SOURCES = {"surface": ("prf-visualrois", "nsdgeneral")}

from lib.datasets import (
    compute_shared_stimuli,
    filter_by_stimulus,
    nsd,
    split_by_repetition,
)
from lib.spectra import (
    compute_cross_individual_spectra,
    compute_within_individual_spectra,
    plot_spectra,
)
from lib.utilities import JOURNAL_MATPLOTLIBRC, mathtext_exponent_label

FIGURES_HOME = Path.cwd().parent / "figures-laion"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

# NEW: settings
SUBJECTS = [0, 1, 2, 3, 4, 5, 6, 7]  
REFERENCE_SUBJECT = 0
PREPROCESSING = "fithrf_GLMdenoise_RR"   # b3, as in LAION
N_IMAGES_WITHIN = 5833                   # LAION's within-subject image count
SEED = 0
N_PERMUTATIONS = 5000                      # 10 for testing, 5000 for the final run

# %% load datasets
datasets = {
    subject: nsd.load_dataset(
        subject=subject,
        roi="general",
        preprocessing=PREPROCESSING,
        z_score=True,
    )
    for subject in SUBJECTS
}

# NEW: within-subject, randomly subsample to LAION's image count
rng = np.random.default_rng(SEED)
datasets_within = {}
for subject, dataset in datasets.items():
    stimuli = compute_shared_stimuli([dataset], n_repetitions=2)
    subset = set(rng.choice(sorted(stimuli), size=N_IMAGES_WITHIN, replace=False))
    print(f"subject {subject + 1}: {len(stimuli)} images available, {len(subset)} used", flush=True)
    datasets_within[subject] = split_by_repetition(
        filter_by_stimulus(dataset, stimuli=subset),
        n_repetitions=2,
    )

# between-subject: unchanged (NSD has fewer shared images than LAION, nothing to subsample)
shared_stimuli = compute_shared_stimuli(datasets.values(), n_repetitions=2)
print(f"shared images: {len(shared_stimuli)}", flush=True)

datasets_cross = {
    subject: split_by_repetition(
        filter_by_stimulus(dataset, stimuli=shared_stimuli),
        n_repetitions=2,
    )
    for subject, dataset in datasets.items()
}

# %% compute spectra
spectra_within = compute_within_individual_spectra(
    datasets_within,
    n_permutations=N_PERMUTATIONS,
)
spectra_cross = compute_cross_individual_spectra(
    datasets_cross,
    reference_individual=REFERENCE_SUBJECT,
    n_permutations=N_PERMUTATIONS,
    randomized=True,
)

# %% plot spectra
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
    hue_order=list(reversed(SUBJECTS)),
    hue_labels=[f"{subject + 1}" for subject in reversed(SUBJECTS)],
    marker="s",
    hide_insignificant=True,
    null_quantile=0.999,
)
ax.set_title(f"within-subject ({N_IMAGES_WITHIN} images)", pad=10)
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
    hue_order=list(reversed(SUBJECTS)),
    hue_labels=[
        f"{subject + 1}*" if subject == REFERENCE_SUBJECT else f"{subject + 1}"
        for subject in reversed(SUBJECTS)
    ],
    marker=None,
    hide_insignificant=True,
    null_quantile=0.999,
)
ax.set_title(f"between-subject,\nrelative to subject {REFERENCE_SUBJECT + 1}")
ax.set_ylabel("cross-covariance")
ax.set_xlabel("rank")
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

fig.suptitle("NSD, b3 betas (fithrf_GLMdenoise_RR), within-subject images matched to LAION")
name = f"general_nsd_b3_matched_perm_{N_PERMUTATIONS}"
fig.savefig(FIGURES_HOME / f"{name}.pdf", bbox_inches="tight")
fig.savefig(FIGURES_HOME / f"{name}.png", dpi=150, bbox_inches="tight")