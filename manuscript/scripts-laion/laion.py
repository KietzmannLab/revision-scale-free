import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from laion_fmri.config import dataset_initialize
from laion_fmri.splits import get_train_test_ids
from laion_fmri.subject import load_subject

IDENTIFIER = "laion_fmri"
DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
CACHE_DIR = Path(os.environ.get("BONNER_CACHING_HOME", "cache"))
SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
N_SUBJECTS = len(SUBJECTS)
RESOLUTION_IN_MM = 1.778

ROIS: Mapping[str, Sequence[str]] = {
    "general": ("laiongeneral",),
    "V1-4": ("V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"),
    "V1": ("V1v", "V1d"),
    "V2": ("V2v", "V2d"),
    "V3": ("V3v", "V3d"),
    "V4": ("hV4",),
    "places": ("OPA", "PPA", "MPA"),
    "faces": ("FFA1", "FFA2", "OFA", "pSTSfaces"),
    "bodies": ("EBA", "FBA"),
    "words": ("VWFA1", "VWFA2", "mfswords", "pSTSwords"),
    "OPA": ("OPA",),
    "PPA": ("PPA",),
    "early visual stream": ("laionEVC",),
    "lateral visual stream": ("laionlateral",),
    "ventral visual stream": ("laionventral",),
    "parietal visual stream": ("laiondorsal",),
}


def _mask_on_voxel_axis(sub, name):
    mask = np.asarray(sub.get_roi_mask(name), dtype=bool)
    if mask.size == sub.get_n_voxels():
        return mask
    brain = np.asarray(sub.get_brain_mask(), dtype=bool)
    if mask.size == brain.size:
        return mask[brain]
    raise ValueError(f"ROI mask {name} has size {mask.size}, matching neither the voxel axis nor the brain grid")


def _roi_mask(sub, roi):
    return np.logical_or.reduce([_mask_on_voxel_axis(sub, name) for name in ROIS[roi]])


def _voxel_coordinates(sub, mask):
    coords = np.asarray(sub.get_voxel_coordinates())[mask]
    if np.allclose(coords, np.round(coords)) and coords.min() >= 0 and coords.max() < 256:
        coords = np.round(coords).astype(np.uint8)
    return coords


def _ood_labels():
    _, test_ids = get_train_test_ids("ood", pool="shared")
    return set(test_ids)


def _load_betas(*, subject, z_score, roi):
    sub = load_subject(SUBJECTS[subject])
    mask = _roi_mask(sub, roi)
    chunks, tables = [], []
    for session in sub.get_sessions():
        betas = np.asarray(
            sub.get_betas(session=session, roi=list(ROIS[roi]), streaming=True),
            dtype=np.float32,
        )
        trials = sub.get_trial_info(session=session).reset_index(drop=True)
        if betas.shape != (len(trials), mask.sum()):
            raise ValueError(
                f"{SUBJECTS[subject]} {session}: betas {betas.shape}, "
                f"trials {len(trials)}, ROI voxels {mask.sum()}",
            )
        if z_score:
            with np.errstate(invalid="ignore", divide="ignore"):
                betas = (betas - np.nanmean(betas, axis=0)) / np.nanstd(betas, axis=0)
        trials["trial"] = np.arange(len(trials))
        chunks.append(betas)
        tables.append(trials)

    betas = np.concatenate(chunks, axis=0)
    trials = pd.concat(tables, ignore_index=True)
    trials["session"] = trials["session"].str.split("-").str[1].astype(int)
    trials = trials.sort_values(["session", "run", "trial"], kind="stable")
    betas = betas[trials.index.to_numpy()]
    trials = trials.reset_index(drop=True)
    trials["repetition"] = trials.groupby("label").cumcount()

    order = np.lexsort((trials["repetition"].to_numpy(), trials["label"].to_numpy()))
    trials = trials.iloc[order].reset_index(drop=True)
    betas = betas[order]

    valid = np.isfinite(betas).all(axis=0)
    coords = _voxel_coordinates(sub, mask)[valid]

    return xr.DataArray(
        data=betas[:, valid],
        dims=("presentation", "neuroid"),
        coords={
            "session": ("presentation", trials["session"].to_numpy(np.uint8)),
            "trial": ("presentation", trials["trial"].to_numpy(np.uint16)),
            "run": ("presentation", trials["run"].to_numpy(np.uint8)),
            "stimulus": ("presentation", trials["label"].to_numpy(str)),
            "repetition": ("presentation", trials["repetition"].to_numpy(np.uint8)),
            "x": ("neuroid", coords[:, 0]),
            "y": ("neuroid", coords[:, 1]),
            "z": ("neuroid", coords[:, 2]),
        },
    )


def _open_betas_by_roi(*, subject, resolution, preprocessing, z_score, roi):
    path = (
        CACHE_DIR
        / "data"
        / f"dataset={IDENTIFIER}"
        / "betas"
        / f"resolution={resolution}"
        / f"preprocessing={preprocessing}"
        / f"z_score={z_score}"
        / f"roi={roi}"
        / f"subject={subject}.pkl"
    )
    if path.exists():
        return pd.read_pickle(path)
    betas = _load_betas(subject=subject, z_score=z_score, roi=roi).assign_attrs({
        "resolution": resolution,
        "preprocessing": preprocessing,
        "z_score": z_score,
        "subject": subject,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(betas, path)
    return betas


def load_dataset(
    *,
    subject: int,
    resolution: Literal["1pt8mm"] = "1pt8mm",
    preprocessing: Literal["fithrf_GLMdenoise_RR"] = "fithrf_GLMdenoise_RR",
    z_score: bool = True,
    roi: str = "general",
    exclude_ood: bool = True,
) -> xr.DataArray:
    if resolution != "1pt8mm":
        raise ValueError("LAION-fMRI only provides betas at 1pt8mm")
    if preprocessing != "fithrf_GLMdenoise_RR":
        raise ValueError("LAION-fMRI only provides fithrf_GLMdenoise_RR (TYPED) betas")
    if roi not in ROIS:
        raise ValueError(f"Unknown ROI {roi}; available: {list(ROIS)}")

    dataset_initialize(DATA_DIR)
    betas = _open_betas_by_roi(
        subject=subject,
        resolution=resolution,
        preprocessing=preprocessing,
        z_score=z_score,
        roi=roi,
    ).assign_attrs({"roi": roi})

    if exclude_ood:
        is_ood = np.isin(betas["stimulus"].data, list(_ood_labels()))
        betas = betas.isel(presentation=~is_ood).assign_attrs({"ood": "excluded"})

    identifier = ".".join([f"{key}={value}" for key, value in betas.attrs.items()])
    return (
        betas.rename(f"{IDENTIFIER}.{identifier}")
        .set_xindex(["stimulus", "repetition"])
        .set_xindex(["x", "y", "z"])
    )


if __name__ == "__main__":
    betas = load_dataset(subject=0, roi="general")
    print(betas)
    rep0 = betas.isel(presentation=betas["repetition"].data == 0)
    rep1 = betas.isel(presentation=betas["repetition"].data == 1)
    print("Repetition 0:", rep0.shape, "Repetition 1:", rep1.shape)
    print("Same stimuli in same order:", (rep0["stimulus"].data == rep1["stimulus"].data).all())
    print("Repetitions per image:", pd.Series(betas["stimulus"].data).value_counts().value_counts().sort_index().to_dict())
