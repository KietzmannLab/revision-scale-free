import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "manuscript/scripts-laion")
import laion as mine
import laion_group as group

a = mine.load_dataset(subject=0, roi="general")
b = group.load_dataset(subject="sub-01", roi="general")
print("mine :", a.shape)
print("group:", b.shape)

a_order = np.lexsort((a["trial"].values, a["session"].values))
b_session = np.array([int(str(s).split("-")[1]) for s in b["session"].values])
b_order = np.lexsort((b["beta_index"].values, b["run"].values, b_session))

if a.shape != b.shape:
    print("Different shapes, cannot compare values.")
    sys.exit()

A = a.values[a_order]
B = b.values[b_order]
print("max abs difference in betas:", float(np.nanmax(np.abs(A - B))))
print("same repetition numbers:", np.array_equal(a["repetition"].values[a_order], b["repetition"].values[b_order]))

pairs = pd.DataFrame({"label": a["stimulus"].values[a_order], "stim_idx": b["stimulus"].values[b_order]})
print("label -> stim_idx is one-to-one:",
      pairs.groupby("label")["stim_idx"].nunique().max() == 1
      and pairs.groupby("stim_idx")["label"].nunique().max() == 1)

coords_a = np.stack([a["x"].values, a["y"].values, a["z"].values], axis=1)
coords_b = np.stack([b["x"].values, b["y"].values, b["z"].values], axis=1)
print("unique voxel coordinates (group):", len(np.unique(coords_b, axis=0)) == len(coords_b))
print("coordinates identical:", np.array_equal(coords_a, coords_b))
