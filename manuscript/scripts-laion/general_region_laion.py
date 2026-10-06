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
# # general-region

# %%
from pathlib import Path

import cairosvg
import seaborn as sns
import skunk
import xarray as xr
from bonner.plotting import save_figure
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

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

FIGURES_HOME = Path.cwd().parent / "figures"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

REFERENCE_SUBJECT = 0

# %% [markdown]
# ## load datasets

# %%
datasets = {
    subject: nsd.load_dataset(
        subject=subject,
        roi="general",
        preprocessing="fithrf",
        z_score=True,
    )
    for subject in range(nsd.N_SUBJECTS)
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

# %% [markdown]
# ## compute spectra

# %%
spectra_within = compute_within_individual_spectra(
    datasets_within,
    n_permutations=5_000,
)
spectra_cross = compute_cross_individual_spectra(
    datasets_cross,
    reference_individual=REFERENCE_SUBJECT,
    n_permutations=5_000,
    randomized=True,
)

# %% [markdown]
# ## plot spectra

# %%
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
    hue_order=list(reversed(range(nsd.N_SUBJECTS))),
    hue_labels=[f"{subject + 1}" for subject in reversed(range(nsd.N_SUBJECTS))],
    marker="s",
    hide_insignificant=True,
    null_quantile=0.999,
)
ax.set_title("within-subject", pad=10)
ax.set_ylabel("covariance")
ax.set_xlabel("rank")
ax.legend(**kwargs_legend)

ax_inset = ax.inset_axes([0.7, 0.65, 0.22, 0.22])
ax_inset.axis("off")
skunk.connect(ax_inset, "human")

ax = axes[1]
plot_spectra(
    ax=ax,
    spectra=spectra_cross,
    hue="individual",
    palette="flare",
    hue_reference=REFERENCE_SUBJECT,
    hue_order=list(reversed(range(nsd.N_SUBJECTS))),
    hue_labels=[
        f"{subject + 1}*" if subject == REFERENCE_SUBJECT else f"{subject + 1}"
        for subject in reversed(range(nsd.N_SUBJECTS))
    ],
    marker=None,
    hide_insignificant=True,
    null_quantile=0.999,
)
ax.set_title(
    f"between-subject,\nrelative to subject {REFERENCE_SUBJECT + 1}",
)
ax.set_ylabel("cross-covariance")
ax.set_xlabel("rank")
ax.axvline(len(shared_stimuli), ls="--", c="gray", lw=0.5, ymax=0.45)
ax.text(
    s="number of\nshared images",
    x=3e3,
    y=2e-9,
    fontsize="xx-small",
    ha="center",
    va="bottom",
)
ax.legend(**kwargs_legend)
ax_inset = ax.inset_axes([0.6, 0.65, 0.23, 0.23])
ax_inset.axis("off")
skunk.connect(ax_inset, "humans")

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

svg = skunk.insert(
    {
        "human": f"{FIGURES_HOME}/human.svg",
        "humans": f"{FIGURES_HOME}/humans.svg",
    },
)
cairosvg.svg2pdf(
    bytestring=svg.encode(),
    write_to=f"{FIGURES_HOME}/general.pdf",
)

# %% [markdown]
# ## compute more spectra

# %%
spectra = {
    reference_subject: compute_cross_individual_spectra(
        datasets_cross,
        reference_individual=reference_subject,
        n_permutations=5_000,
        randomized=True,
    )
    for reference_subject in tqdm(
        range(nsd.N_SUBJECTS),
        desc="reference individual",
        leave=False,
    )
}

# %% [markdown]
# ## plot all cross-individual spectra with each subject as the reference subject

# %%
palette = sns.color_palette("flare", nsd.N_SUBJECTS)

ytick_exponents = list(range(-7, 0))

fig, axes = plt.subplots(figsize=(5, 6), ncols=3, nrows=3, sharex=True, sharey=True)
for reference_subject, ax in zip(range(nsd.N_SUBJECTS), axes.flat, strict=False):
    plot_spectra(
        spectra[reference_subject],
        ax=ax,
        hue="individual",
        hue_reference=reference_subject,
        hue_order=list(reversed(range(nsd.N_SUBJECTS))),
        hue_labels=[
            f"{subject + 1}*" if subject == reference_subject else f"{subject + 1}"
            for subject in reversed(range(nsd.N_SUBJECTS))
        ],
        palette=palette,
        marker=None,
        hide_insignificant=True,
        null_quantile=0.999,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"subject {reference_subject + 1}")
    ax.set_xlim(left=1, right=1e3)
    ax.set_ylim(bottom=1e-7, top=1e-1)
    ax.set_yticks(
        [10**exponent for exponent in ytick_exponents],
        labels=[
            mathtext_exponent_label(exponent) if exponent % 2 == 1 else ""
            for exponent in ytick_exponents
        ],
    )

# mean across comparisons
ax = axes.flat[-1]
spectra_mean = []
for reference_individual in range(8):
    spectrum = (
        spectra[reference_individual]
        .drop_indexes("individual")
        .isel(
            individual=[
                individual
                for individual in range(8)
                if individual != reference_individual
            ],
        )
        .rename({"individual": "comparison"})
        .assign_coords({
            "comparison": [
                f"{reference_individual}-{individual}"
                for individual in range(8)
                if individual != reference_individual
            ]
        })
    )
    spectra_mean.append(spectrum)

spectra_mean = xr.concat(spectra_mean, dim="comparison").mean("comparison")

mean = spectra_mean.mean("fold")
threshold = 0.001
p_values = (mean["covariance"] < mean["covariance (permuted)"]).mean("permutation")
significant = (p_values < threshold).to_numpy()

kwargs_significant = {
    "ls": "None",
    "c": "k",
    "marker": "o",
    "zorder": 2,
    "mew": 0,
    "alpha": 1,
}
kwargs_insignificant = {
    "mew": 1,
    "alpha": 0.5,
    "mfc": "None",
}

ax.errorbar(
    spectra_mean["rank"][significant],
    spectra_mean["covariance"].mean("fold")[significant],
    spectra_mean["covariance"].std("fold")[significant],
    **kwargs_significant,
)
ax.errorbar(
    spectra_mean["rank"][~significant],
    spectra_mean["covariance"].mean("fold")[~significant],
    spectra_mean["covariance"].std("fold")[~significant],
    **kwargs_significant | kwargs_insignificant,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("mean across comparisons")

axes[0, 0].legend(
    **kwargs_legend
    | {
        "loc": "lower left",
        "borderpad": 0.1,
        "borderaxespad": 0,
        "columnspacing": 0.025,
        "handletextpad": 0.025,
        "labelspacing": 0.45,
    },
)

fig.supxlabel("rank", y=0.025, x=0.57)
fig.supylabel("cross-covariance", x=0.03)
fig.suptitle("between-subject, relative to ...", x=0.57)
fig.tight_layout()

save_figure(fig, filepath=FIGURES_HOME / "general-all.pdf")
