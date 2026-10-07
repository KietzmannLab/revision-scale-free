"""Loader for LAION-fMRI, producing the same data format as nsd.py."""

import xarray as xr
import pandas as pd
import numpy as np

from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import get_subjects
from laion_fmri.subject import load_subject
from bonner.caching import cache

# --- Constants ---------------------------------------------------------
IDENTIFIER = "laion"
SUBJECTS = ['sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07']
DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
ROIS: dict[str, list[str]] = {
    "general": ["laiongeneral"],   
    "V1-4": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "V1": ["V1v", "V1d"],
    "V2": ["V2v", "V2d"],
    "V3": ["V3v", "V3d"],
    "V4": ["hV4"],
}
# --- 1. One session ----------------------------------------------------
def _load_session(subject, session, roi):
    """Betas of one session as a labeled, z-scored xarray."""

    # get betas + trial table
    betas = subject.get_betas(session=session, roi=ROIS[roi], streaming=True)
    trials = subject.metadata
    trials = trials[trials["session"] == session].sort_values(["run", "beta_index"]) # sort by run and within run by betaindex to make sure the variables "trails" and "betas" match per row

    # build xarray with dims (presentation, neuroid) + stimulus label 
    data = xr.DataArray(
        data=betas,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus":   ("presentation", trials["stim_idx"].to_numpy()),
            "session":    ("presentation", trials["session"].to_numpy()),
            "dataset":    ("presentation", trials["dataset"].to_numpy()),
            "beta_index": ("presentation", trials["beta_index"].to_numpy()),
            "run": ("presentation", trials["run"].to_numpy()),
        },
    )
    # z-score each voxel within this session
    data_z = (data - data.mean("presentation")) / data.std("presentation")

    return data_z

# --- 2. All sessions ---------------------------------------------------
def _load_all_sessions(subject, roi, sessions=None):
    """Stack all sessions of one subject."""
    
    # use all sessions if nothing is passed, otherwise use passed subset
    if sessions is None: 
        sessions = subject.get_sessions()

    # loop over sessions, concatenate
    sessions_list = [_load_session(subject, session, roi) for session in sessions]
    data = xr.concat(sessions_list, dim="presentation") # concatenate by row, since more trials are stacked together not voxels 

    coords = subject.get_voxel_coordinates(roi=ROIS[roi])   # (n_voxels, 3), millimetres
    grid = np.round(coords / 1.778).astype(int)              # clean integer positions
    data = data.assign_coords(
        x=("neuroid", grid[:, 0]),
        y=("neuroid", grid[:, 1]),
        z=("neuroid", grid[:, 2]),
    )

    # add repetition index (0, 1, 2, ... per image, in time order)
    labels = pd.DataFrame({"stimulus": data["stimulus"].values}) # put row labels into table to count stimulus-ID occurrences 
    repetition = labels.groupby("stimulus").cumcount().to_numpy() # groups images with same ID and numbers the rows wtihin each group
    data = data.assign_coords(repetition=("presentation", repetition))
    
    # drop OOD images
    keep_rows = (data["dataset"] != "OOD").values
    data = data.isel(presentation=keep_rows)  

    # drop voxels that are NaN in any session
    has_nan = data.isnull().any("presentation").values   # True for voxels with any NaN
    data = data.isel(neuroid=~has_nan)                    # keep voxels WITHOUT NaN

    return data


# --- small cache wrapper -------------------------------------
@cache("data/dataset=laion/betas=TYPED/z_score=True/roi={roi}/subject={subject}.nc")
def _load_all_sessions_cached(*, subject, roi):
    """All sessions of one subject; cached on disk after the first call."""
    dataset_initialize(DATA_DIR)
    sub = load_subject(subject)
    return _load_all_sessions(sub, roi).rename("betas")

# --- 3. Public entry point --------------------------------------------
def load_dataset(*, subject, roi="general", sessions=None):
    """Same role as nsd.load_dataset: returns the final labeled array."""

    # load the data: full data from the cache, test subsets without caching
    if sessions is None:
        data = _load_all_sessions_cached(subject=subject, roi=roi)
    else:
        dataset_initialize(DATA_DIR)
        sub = load_subject(subject)              # name → object
        data = _load_all_sessions(sub, roi, sessions=sessions)

    # assign further attributes so they match nsd signature
    attrs = {"betas": "TYPED", "z_score": "True", "subject": subject, "roi": roi}
    if sessions is not None:
        attrs["sessions"] = "-".join(sessions)
    data = data.assign_attrs(attrs)

    # build name from configurations to use for caching
    identifier = ".".join([f"{key}={value}" for key, value in data.attrs.items()])
    data = data.rename(f"{IDENTIFIER}.{identifier}")

    # index rows by (stimulus, repetition) and voxels by (x, y, z), as in nsd.py
    data = data.set_xindex(["stimulus", "repetition"]).set_xindex(["x", "y", "z"])

    return data

