from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject

DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"

dataset_initialize(DATA_DIR)

print("Subjects:", get_subjects())
describe()

subject = "sub-01"
session = "ses-01"

print(f"\nFace ROIs for {subject}:", get_rois(subject, category="face"))

sub = load_subject(subject)

betas = sub.get_betas(session=session)
print(f"\nBetas for {subject} {session}:", getattr(betas, "shape", type(betas)))

trials = sub.get_trial_info(session=session)
print(f"\nTrials in {session}: {len(trials)}")
print(trials.head())

print("\nLoader test finished.")
