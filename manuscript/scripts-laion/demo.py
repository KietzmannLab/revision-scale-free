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
# # Demo
#
# This demo shows how to run our cross-decomposition analyses using a high-level API. If you're interested in the low-level details of how this works, feel free to check out the source code!

# %%

import matplotlib
matplotlib.use("Agg")

import tempfile
import warnings
from pathlib import Path

import seaborn as sns
import xarray as xr
from bonner.plotting import DEFAULT_MATPLOTLIBRC
from matplotlib import pyplot as plt
from osfclient.api import OSF

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

sns.set_theme(context="paper", style="ticks", rc=DEFAULT_MATPLOTLIBRC)
warnings.filterwarnings("ignore")


# %% [markdown]
# ## Prepare the datasets (~300 MB download)
#
# Here, we analyze a small subset of the Natural Scenes Dataset containing fMRI responses from the primary visual cortex (V1) to 1,000 natural images seen twice by each of two individuals.
#
# We inspect the data for one individual, which shows the matrix of fMRI activations to 1,000 natural images (along the `presentation` dimension) from 1,350 voxels (along the `neuroid` dimension) along with the associated metadata.

# %%
def load_toy_datasets() -> dict[int, dict[int, xr.DataArray]]:
    datasets = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        project_id = "ft8b5"

        storage = OSF().project(project_id=project_id).storage("osfstorage")

        for file_ in storage.files:
            filepath = Path(tmpdir) / file_.path.lstrip("/")
            filepath.parent.mkdir(exist_ok=True, parents=True)
            with open(filepath, mode="wb") as f:
                file_.write_to(f)

            dataset = xr.load_dataarray(filepath).assign_attrs({"roi": "V1"})
            identifier = ".".join([
                f"{key}={value}" for key, value in dataset.attrs.items()
            ])
            datasets[dataset.attrs["subject"]] = (
                dataset.rename(f"{nsd.IDENTIFIER}.{identifier}")
                .set_xindex(["stimulus", "repetition"])
                .set_xindex(["x", "y", "z"])
            )

    # filter both datasets to extract only responses to the 1,000 shared images
    # split each of them by repetition, i.e. the responses on trials 1 and 2.
    shared_stimuli = compute_shared_stimuli(datasets.values(), n_repetitions=2)

    return {
        subject: split_by_repetition(
            filter_by_stimulus(dataset, stimuli=shared_stimuli),
            n_repetitions=2,
        )
        for subject, dataset in datasets.items()
    }


datasets = load_toy_datasets()
print(datasets[0][0])

# %% [markdown]
# ## Cross-decomposition analysis
#
# Given two datasets $X \in \mathbb{R}^{n \times d_X}$ and $Y \in \mathbb{R}^{n \times d_Y}$ containing fMRI activations to $n$ stimuli from $d_X$, $d_Y$ voxels, we identify the latent dimensions shared between $X$ and $Y$ on a training subset of stimuli,
#
# $$
# \begin{align*}
#     \text{cov} \left(X_\text{train}, Y_\text{train}\right)
#     &= \dfrac{1}{n_\text{train}} X_\text{train}^\top Y_\text{train}\\
#     &= U \Sigma V^\top
# \end{align*}
# $$
#
# and evaluate the stimulus-related variance by projecting held-out test data on the learnt latent dimensions:
#
# $$
# \begin{align*}
#     \Sigma_\text{test}
#     &= \text{cov} \left(X_\text{test} U, Y_\text{test} V\right)\\
#     &= \dfrac{1}{n_\text{test}} \left(X_\text{test} U \right)^\top \left(Y_\text{test} V\right) \;.\\
# \end{align*}
# $$
#
# We extract the diagonal of this matrix, bin it logarithmically across ranks, and average the spectra across 8 cross-validation folds.

# %% [markdown]
# ### Computing within-individual covariance spectra
#
# When we compute within-individual covariance spectra, we measure stimulus-related variance that generalizes across multiple presentations of the same stimuli within an individual, i.e. $X$ and $Y$ are data matrices from the same individual on different trials.

# %%
spectra = compute_within_individual_spectra(
    datasets,
    n_folds=8,
)

fig, ax = plt.subplots()
plot_spectra(
    spectra,
    ax=ax,
    hue="individual",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(left=1, right=2e3)
ax.set_ylim(top=1e-1, bottom=1e-7)
ax.legend(title="subject")
ax.set_title("within-individual covariance spectra")
ax.set_xlabel("rank")
ax.set_ylabel("covariance")
fig.savefig("demo_within.png")

# %% [markdown]
# ### Computing cross-individual covariance spectra
#
# When we compute cross-individual covariance spectra, we measure stimulus-related variance that generalizes across different individuals, i.e. $X$ and $Y$ are data matrices from different individuals.

# %%
reference_subject = 0

spectra = compute_cross_individual_spectra(
    datasets,
    reference_individual=reference_subject,
    n_folds=8,
)

fig, ax = plt.subplots()
plot_spectra(
    spectra,
    ax=ax,
    hue="individual",
    hue_reference=f"{1 + reference_subject}",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(left=1, right=2e3)
ax.set_ylim(top=1e-1, bottom=1e-7)
ax.legend(title="subject")
ax.set_title(
    f"cross-individual covariance spectra,\nrelative to subject {1 + reference_subject}",
)
ax.set_xlabel("rank")
ax.set_ylabel("covariance")
fig.savefig("demo_between.png")

# %% [markdown]
# ## Scale-free power law structure
#
# Even in this simple demonstration, we see that the covariance spectra have a scale-free power-law structure, both within an individual and shared between individuals. In our paper, we go into much more detail about what this means, so check it out!
