import numpy as np

from lib.datasets import laion
from lib.datasets.laion_mni import (
    build_projection,
    check_against_laion_to_template,
    downsample_indices,
    load_mni305_reference,
)

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

datasets = {s: laion.load_dataset(subject=s, roi="general") for s in SUBJECTS}
reference = load_mni305_reference()
source, _ = downsample_indices(reference.shape[:3], resolution=1.8)
source = source.ravel()

support = {}
for s, d in datasets.items():
    weights = build_projection(d, subject=s, reference=reference, order=1)
    support[s] = np.diff(weights.indptr)[source] > 0
    print(
        s,
        "native:", d.sizes["neuroid"],
        "MNI 1.8mm:", int(support[s].sum()),
        "check vs to_template:", check_against_laion_to_template(d, subject=s),
        flush=True,
    )

print("pairwise shared voxels")
for a in SUBJECTS:
    print(a, [int((support[a] & support[b]).sum()) for b in SUBJECTS])
print("all subjects:", int(np.logical_and.reduce(list(support.values())).sum()))