import time
from lib.datasets.laion import load_dataset

SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ROIS = ["general", "V1", "V2", "V3", "V4"]

for subject in SUBJECTS:
    for roi in ROIS:
        t0 = time.time()
        X = load_dataset(subject=subject, roi=roi)
        print(f"{subject} {roi}: {X.shape}, {time.time() - t0:.0f} s", flush=True)