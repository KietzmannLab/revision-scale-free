from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from lib.datasets import compute_shared_stimuli, filter_by_stimulus, laion, split_by_repetition
from lib.spectra import compute_within_individual_spectra

SUBJECTS = [0]
N_PERMUTATIONS = 0
OUTPUT = Path("results/laion/general-region")
OUTPUT.mkdir(parents=True, exist_ok=True)
LABEL = "subj" + "-".join(str(subject + 1) for subject in SUBJECTS)

datasets = {
    subject: laion.load_dataset(subject=subject, roi="general")
    for subject in SUBJECTS
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

for subject, dataset in datasets_within.items():
    print(
        f"subject {subject}: rep0 {dataset[0].shape}, rep1 {dataset[1].shape}, "
        f"aligned={(dataset[0]['stimulus'].data == dataset[1]['stimulus'].data).all()}",
        flush=True,
    )

spectra_within = compute_within_individual_spectra(
    datasets_within,
    n_permutations=N_PERMUTATIONS,
)
spectra_within.to_netcdf(OUTPUT / f"spectra_within_{LABEL}.nc")

fig, ax = plt.subplots(figsize=(3.5, 3))
colors = plt.cm.viridis(np.linspace(0, 0.9, laion.N_SUBJECTS))
rank = spectra_within["rank"].to_numpy()
for subject in SUBJECTS:
    covariance = spectra_within["covariance"].sel(individual=subject)
    mean = covariance.mean("fold").to_numpy()
    std = covariance.std("fold").to_numpy()
    positive = mean > 0
    ax.errorbar(
        rank[positive],
        mean[positive],
        std[positive],
        ls="None",
        marker="s",
        ms=3,
        c=colors[subject],
        label=f"{subject + 1}",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(left=1, right=1e4)
ax.set_xlabel("rank")
ax.set_ylabel("covariance")
ax.set_title("within-subject (LAION-fMRI)")
ax.legend(title="subject", loc="lower left", ncols=2)
fig.tight_layout()
fig.savefig(OUTPUT / f"general-within_{LABEL}.pdf")
fig.savefig(OUTPUT / f"general-within_{LABEL}.png", dpi=200)
print("Saved", OUTPUT, flush=True)