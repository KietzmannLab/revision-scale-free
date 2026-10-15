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
# # spectra-visual

# %%
import matplotlib
matplotlib.use("Agg")

from collections.abc import Collection
from pathlib import Path


import numpy as np
import seaborn as sns
import xarray as xr
from bonner.plotting import save_figure
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm.auto import tqdm

from lib.datasets import (
    compute_shared_stimuli,
    filter_by_stimulus,
    laion,
    split_by_repetition,
)
from lib.spectra import (
    compute_cross_individual_spectra,
    compute_cross_roi_spectra,
    compute_within_individual_spectra,
    plot_spectra,
)
from lib.utilities import (
    JOURNAL_MATPLOTLIBRC,
    mathtext_exponent_label,
)

FIGURES_HOME = Path.cwd().parent / "figures-laion"
FIGURES_HOME.mkdir(exist_ok=True, parents=True)

sns.set_theme(context="paper", style="ticks", rc=JOURNAL_MATPLOTLIBRC)

REFERENCE_SUBJECT = 0

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ROIS_VISUAL = ("V1", "V2", "V3", "V4")
ROIS_ALL = ROIS_VISUAL
PALETTE = dict(
    zip(
        ROIS_VISUAL,
        sns.cubehelix_palette(
            start=2.75,
            rot=0,
            dark=0.25,
            light=0.75,
            reverse=True,
            n_colors=len(ROIS_VISUAL),
        ),
        strict=True,
    ),
) | {"frontal": "gray"}

# %% [markdown]
# ## load datasets

# %%
datasets = {
    roi: {
        subject: laion.load_dataset(
            subject=s,
            roi=roi,
        )
        for subject, s in enumerate(SUBJECTS)
    }
    for roi in tqdm(ROIS_ALL, desc="region of interest", leave=False)
}

datasets_within = {
    roi: {
        subject: split_by_repetition(
            filter_by_stimulus(
                dataset,
                stimuli=compute_shared_stimuli([dataset], n_repetitions=2),
            ),
            n_repetitions=2,
        )
        for subject, dataset in tqdm(datasets_.items(), desc="subject", leave=False)
    }
    for roi, datasets_ in tqdm(datasets.items(), desc="region of interest", leave=False)
}

shared_stimuli = compute_shared_stimuli(datasets["V1"].values(), n_repetitions=2)

datasets_cross = {
    roi: {
        subject: split_by_repetition(
            filter_by_stimulus(dataset, stimuli=shared_stimuli),
            n_repetitions=2,
        )
        for subject, dataset in tqdm(datasets_.items(), desc="subject", leave=False)
    }
    for roi, datasets_ in tqdm(datasets.items(), desc="region of interest", leave=False)
}

# %% [markdown]
# ## compute spectra

# %%
spectra_within = {
    roi: compute_within_individual_spectra(
        datasets_within[roi],
        n_permutations=5000,
    ).expand_dims({"region of interest": [roi]})
    for roi in tqdm(ROIS_ALL, desc="roi", leave=False)
}

spectra_cross = {
    roi: compute_cross_individual_spectra(
        datasets_cross[roi],
        reference_individual=REFERENCE_SUBJECT,
        n_permutations=5000,
        randomized=True,
    ).expand_dims({"region of interest": [roi]})
    for roi in tqdm(ROIS_ALL, desc="roi", leave=False)
}

# %% [markdown]
# ## plot visual-regions

# %%
fig = plt.figure(figsize=(6, 3.75), layout="constrained")
subfigures = fig.subfigures(ncols=2, wspace=0.05)
axes_within = subfigures[0].subplots(nrows=2, ncols=2, sharex=True, sharey=True)
axes_cross = subfigures[1].subplots(nrows=2, ncols=2, sharex=True, sharey=True)

for roi, axes_within_, axes_cross_ in zip(
    tqdm(ROIS_VISUAL, desc="region of interest", leave=False),
    axes_within.flat,
    axes_cross.flat,
    strict=True,
):
    plot_spectra(
        spectra=spectra_within[roi],
        ax=axes_within_,
        palette="crest",
        hue="individual",
        hue_order=list(reversed(range(len(SUBJECTS)))),
        hue_labels=[SUBJECTS[subject] for subject in reversed(range(len(SUBJECTS)))],
        marker="s",
        hide_insignificant=True,
        null_quantile=0.99,
    )
    axes_within_.set_title(roi)

    plot_spectra(
        spectra=spectra_cross[roi],
        ax=axes_cross_,
        palette="flare",
        hue="individual",
        hue_reference=REFERENCE_SUBJECT,
        hue_order=list(reversed(range(len(SUBJECTS)))),
        hue_labels=[
            f"{SUBJECTS[subject]}*" if subject == REFERENCE_SUBJECT else SUBJECTS[subject]
            for subject in reversed(range(len(SUBJECTS)))
        ],
        marker=None,
        hide_insignificant=True,
        null_quantile=0.99,
    )
    axes_cross_.set_title(roi)

for ax in (axes_within[1, 1], axes_cross[1, 1]):
    _ = ax.legend(
        loc="upper right",
        title="subject",
        ncols=2,
        columnspacing=0,
        handletextpad=0.0,
        borderpad=0,
        labelspacing=0.35,
        borderaxespad=0,
        reverse=True,
    )

# ax_inset = axes_within[0, 0].inset_axes([0.52, 0.7, 0.23, 0.23])
# ax_inset.axis("off")
# skunk.connect(ax_inset, "human")

# ax_inset = axes_cross[0, 0].inset_axes([0.5, 0.7, 0.3, 0.3])
# ax_inset.axis("off")
# skunk.connect(ax_inset, "humans")

for ax in [*axes_within.flat, *axes_cross.flat]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1, right=1.6e3)
    ax.set_xticks([1, 1e1, 1e2, 1e3])
    ax.set_ylim(bottom=1e-7, top=1e-1)
    ytick_exponents = list(range(-7, 0))
    ax.set_yticks(
        [10**exponent for exponent in ytick_exponents],
        labels=[
            mathtext_exponent_label(exponent) if exponent % 2 == 1 else ""
            for exponent in ytick_exponents
        ],
    )

subfigures[0].supylabel("covariance", ha="center", y=0.5)
subfigures[1].supylabel(
    "cross-covariance",
    ha="center",
)
subfigures[0].suptitle("within-subject", ha="center")
subfigures[1].suptitle(
    f"between-subject, relative to {SUBJECTS[REFERENCE_SUBJECT]}",
    ha="center",
)

for subfigure in subfigures:
    subfigure.supxlabel("rank", x=0.57)


fig.savefig(FIGURES_HOME / "visual_regions_laion_perm=5000.pdf")
fig.savefig(FIGURES_HOME / "visual_regions_laion_perm=5000.png", dpi=150)


# %% [markdown]
# ## plot between-region-spectra

# %%
# def _get_rois(
#     roi: str,
#     *,
#     rois: Collection[str] = ("V1", "V2", "V3", "V4"),
# ) -> list[str]:
#     flag = False
#     output = []
#     for roi_ in rois:
#         if roi_ == roi:
#             flag = True
#         if flag:
#             output.append(roi_)
#     return output


# fig, axes = plt.subplots(
#     ncols=4,
#     figsize=(6, 2.75),
#     sharey=True,
#     sharex=True,
# )

# for roi, ax in zip(
#     tqdm(ROIS_VISUAL, desc="region of interest", leave=False),
#     axes.flat,
#     strict=True,
# ):
#     spectra = compute_cross_roi_spectra(
#         datasets={
#             roi: datasets_within[roi][REFERENCE_SUBJECT]
#             for roi in [*_get_rois(roi), "frontal"]
#         },
#         reference_roi=roi,
#         n_permutations=5_000,
#     )
#     plot_spectra(
#         ax=ax,
#         spectra=spectra,
#         hue="roi",
#         hue_reference=roi,
#         hue_labels=[*_get_rois(roi), "frontal"],
#         hide_insignificant=True,
#         null_quantile=0.999,
#         palette=[PALETTE[roi] for roi in [*_get_rois(roi), "frontal"]],
#     )

#     ax.legend(
#         loc="upper right",
#         borderaxespad=0,
#         fontsize="xx-small",
#         borderpad=0,
#         handletextpad=0.5,
#         labelspacing=0.25,
#     )
#     ax.set_xlim(left=1, right=1.5e2)
#     ax.set_ylim(bottom=1e-5, top=1e-1)
#     ax.set_xticks([1, 1e1, 1e2])
#     ax.set_xscale("log")
#     ax.set_yscale("log")

# axes[0].set_ylabel("covariance")

# fig.suptitle("between-region comparison for subject 1")
# fig.supxlabel("rank", x=0.54, y=0.05)
# save_figure(
#     fig,
#     filepath=FIGURES_HOME / "between-region-spectra.pdf",
# )

# # %% [markdown]
# # ## plot between-region-heatmaps

# # %%
# fig = plt.figure(figsize=(6, 2.5), layout="constrained")
# subfigs = fig.subfigures(ncols=2, width_ratios=[2, 5])

# ax = subfigs[0].subplots()

# plot_spectra(
#     spectra=xr.concat(
#         [
#             compute_within_individual_spectra(
#                 {REFERENCE_SUBJECT: datasets_within[roi][REFERENCE_SUBJECT]},
#                 n_permutations=5_000,
#             ).expand_dims({"region of interest": [roi]})
#             for roi in tqdm(ROIS_VISUAL, desc="region of interest", leave=False)
#         ],
#         dim="region of interest",
#     ),
#     ax=ax,
#     hue="region of interest",
#     hue_order=list(reversed(ROIS_VISUAL)),
#     palette=[PALETTE[roi] for roi in reversed(ROIS_VISUAL)],
#     hide_insignificant=True,
#     null_quantile=0.999,
# )
# ax.legend(
#     loc="upper right",
#     title="",
#     borderaxespad=0,
#     labelspacing=0.25,
#     reverse=True,
# )

# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_ylim(bottom=1e-7, top=1e-1)
# ytick_exponents = list(range(-7, 0))
# ax.set_yticks(
#     [10**exponent for exponent in ytick_exponents],
#     labels=[
#         mathtext_exponent_label(exponent) if exponent % 2 == 1 else ""
#         for exponent in ytick_exponents
#     ],
# )
# ax.set_xlim(left=1, right=2e3)
# ax.set_xticks([1, 1e1, 1e2, 1e3])
# ax.set_xlabel("rank")
# ax.set_ylabel("covariance")
# ax.set_title("within-region", pad=10)

# ax.set_position(Bbox([[0.125, 0.14], [0.9, 0.85]]))

# axes = subfigs[1].subplots(ncols=3)

# spectra = xr.concat(
#     [
#         (
#             compute_cross_roi_spectra(
#                 datasets={
#                     roi: datasets_within[roi][REFERENCE_SUBJECT]
#                     for roi in [*_get_rois(roi), "frontal"]
#                 },
#                 reference_roi=roi,
#                 wide_bins=True,
#             )
#             .rename({"roi": "roi_1"})
#             .expand_dims({"roi_2": [roi]})
#             .mean("fold")
#         )
#         for roi in ROIS_ALL
#     ],
#     dim="roi_2",
# )

# for i_rank, ax in zip(range(spectra.sizes["rank"]), axes.flat, strict=True):
#     spectra_ = (
#         spectra.isel(rank=i_rank)["covariance"]
#         .to_dataframe()
#         .reset_index()
#         .drop(columns="rank")
#         .pivot(columns="roi_2", index="roi_1", values="covariance")
#     )

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="5%", pad=0.5)
#     match i_rank:
#         case 0:
#             ticks = [-3, -2]
#         case 1:
#             ticks = [-5, -4, -3]
#         case 2:
#             ticks = [-6, -5]
#         case _:
#             raise ValueError

#     vmax = None
#     sns.heatmap(
#         ax=axes[i_rank],
#         data=np.log10(spectra_),
#         square=True,
#         vmax=np.max(np.log10(spectra_)) + 0.2,
#         cmap="gist_earth_r",
#         cbar=True,
#         cbar_ax=cax,
#         cbar_kws={"orientation": "horizontal"},
#     )

#     cax.set_xticks(ticks, [mathtext_exponent_label(tick) for tick in ticks])

#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_title(
#         f"rank {mathtext_exponent_label(i_rank)}"
#         f" to {mathtext_exponent_label(i_rank + 1)}",
#     )
#     ax.tick_params(length=0)

#     cax.tick_params(length=0)
#     cax.set_yticks([])
#     if i_rank == 1:
#         cax.set_xlabel("covariance")

# subfigs[1].suptitle("between-region")
# fig.suptitle(f"within-subject, subject {1 + REFERENCE_SUBJECT}")
# save_figure(fig, filepath=FIGURES_HOME / "between-region-heatmaps.pdf")
