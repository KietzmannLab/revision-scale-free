from pathlib import Path

from lib.datasets import compute_shared_stimuli, filter_by_stimulus, laion, split_by_repetition
from laion_mni305 import convert_to_mni305, save_mni305

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
MNI_HOME = Path(__file__).resolve().parent.parent / "results" / "mni305"
datasets = {i: laion.load_dataset(subject=s, roi="general") for i, s in enumerate(SUBJECTS)}
shared_stimuli = compute_shared_stimuli(datasets.values(), n_repetitions=2)
datasets_cross = {
    i: split_by_repetition(filter_by_stimulus(d, stimuli=shared_stimuli), n_repetitions=2)
    for i, d in datasets.items()
}

datasets_mni = convert_to_mni305(datasets_cross, subjects=SUBJECTS)
save_mni305(datasets_mni, subjects=SUBJECTS, directory=MNI_HOME)

print(f"{len(shared_stimuli)} shared stimuli, {datasets_mni[0][0].sizes['neuroid']} shared MNI voxels")
print(f"saved to {MNI_HOME}")
