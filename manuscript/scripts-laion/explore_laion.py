from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import get_subjects
from laion_fmri.subject import load_subject

DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)

print("Subjects:", get_subjects())

sub = load_subject("sub-01")
print("Sessions:", sub.get_sessions())
print("Retinotopy ROIs:", sub.get_available_rois(category="retinotopy"))

# --- one session, V1 (dorsal + ventral) ---
session = "ses-01"
betas = sub.get_betas(session=session, roi=["V1v", "V1d"], streaming=True)
trials = sub.get_trial_info(session=session)

print("\nBetas shape (trials x voxels):", betas.shape)
print("NaN voxels:", int((~(betas == betas)).any(axis=0).sum()))
print("Value range:", float(betas[betas == betas].min()), float(betas[betas == betas].max()))

print("\nTrial table:", trials.shape)
print(trials.columns.tolist())
print(trials.head())

# --- trial table across ALL sessions, with stimulus info ---
meta = sub.metadata
print("\nFull trial table:", meta.shape)
print(meta.columns.tolist())
print(meta.head())
print(meta["dataset"].value_counts())
print(meta["unique_or_shared"].value_counts())

# --- voxel positions ---
coords = sub.get_voxel_coordinates(roi=["V1v", "V1d"])
print("\nVoxel coordinates:", type(coords), getattr(coords, "shape", None))
print(coords[:5] if hasattr(coords, "__getitem__") else coords)