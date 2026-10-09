import numpy as np
from laion_fmri.config import dataset_initialize
from laion_fmri.subject import load_subject
from templateflow.api import get as tflow_get
import nibabel as nib

from lib.datasets import laion
from lib.datasets.laion import DATA_DIR
from laion_mni305 import MNI_RESOLUTION, _brain_indices_of_neuroids, compute_mni305_index_map

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

dataset_initialize(DATA_DIR)
template_path = tflow_get("MNI305", suffix="T1w", extension=".nii.gz")
template = nib.load(str(template_path[0] if isinstance(template_path, list) else template_path))
grid_affine = template.affine @ np.diag([MNI_RESOLUTION] * 3 + [1])
print(f"MNI305 template shape {template.shape}, voxel size {template.header.get_zooms()}")

in_roi = {}
for subject_id in SUBJECTS:
    subject = load_subject(subject_id)
    data = laion.load_dataset(subject=subject_id, roi="general")
    n_brain = subject.get_n_voxels()
    n_roi_total = int(subject.get_roi_mask(laion.ROIS["general"]).sum())
    brain_indices = _brain_indices_of_neuroids(subject, data)

    index = compute_mni305_index_map(subject)
    roi_lookup = np.zeros(n_brain, dtype=bool)
    roi_lookup[brain_indices] = True
    in_brain = index >= 0
    in_roi[subject_id] = in_brain & roi_lookup[np.maximum(index, 0)]

    centroid = (grid_affine @ np.append(np.argwhere(in_roi[subject_id]).mean(0), 1))[:3]
    print(
        f"{subject_id}: brain {n_brain}, ROI {n_roi_total}, loaded (non-NaN) {len(brain_indices)}, "
        f"MNI grid voxels in brain {int(in_brain.sum())}, in loaded ROI {int(in_roi[subject_id].sum())}, "
        f"ROI centroid in MNI mm {np.round(centroid, 1)}"
    )

reference = in_roi[SUBJECTS[0]]
for subject_id in SUBJECTS[1:]:
    print(f"overlap {SUBJECTS[0]} & {subject_id}: {int((reference & in_roi[subject_id]).sum())}")
print(f"overlap all: {int(np.logical_and.reduce(list(in_roi.values())).sum())}")
