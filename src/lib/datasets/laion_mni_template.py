from collections.abc import Mapping

import numpy as np
import torch
import xarray as xr
from laion_fmri.config import dataset_initialize
from laion_fmri.subject import load_subject
from torch.nn.functional import interpolate
from tqdm.auto import tqdm

from lib.datasets.laion import DATA_DIR, SUBJECTS

GRID_SPACING = 1.778


def brain_mask_positions(sub, data: xr.DataArray) -> tuple[np.ndarray, int]:
    coords = sub.get_voxel_coordinates()
    grid = np.round(coords / GRID_SPACING).astype(int)
    position = {tuple(g): i for i, g in enumerate(grid)}
    keys = zip(data["x"].values, data["y"].values, data["z"].values, strict=True)
    return np.array([position[(int(x), int(y), int(z))] for x, y, z in keys]), len(grid)


def to_mni305(
    sub,
    values: np.ndarray,
    *,
    positions: np.ndarray,
    n_brain: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    full = np.zeros((len(values), n_brain), dtype=np.float32)
    full[:, positions] = values
    volume = np.asarray(sub.to_template(full, "MNI305").dataobj)
    return volume.reshape(-1, len(values)).T, volume.shape[:3]


def downsample_indices(
    shape: tuple[int, int, int],
    *,
    resolution: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    linear = torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(1, 1, *shape)
    downsampled = interpolate(linear, scale_factor=1 / resolution).squeeze()
    return downsampled.numpy().astype(np.int64).ravel(), tuple(downsampled.shape)


def convert_datasets_to_mni_space(
    datasets: Mapping[int, Mapping[int, xr.DataArray]],
    /,
    *,
    subject_ids: Mapping[int, str] | None = None,
    resolution: float = 1.8,
    batch_size: int = 50,
) -> dict[int, dict[int, xr.DataArray]]:
    if subject_ids is None:
        subject_ids = dict(enumerate(SUBJECTS))

    dataset_initialize(DATA_DIR)
    subjects = {s: load_subject(subject_ids[s]) for s in datasets}
    positions = {s: brain_mask_positions(subjects[s], datasets[s][0]) for s in datasets}

    inside_roi = {}
    for s in datasets:
        pos, n_brain = positions[s]
        roi, shape = to_mni305(
            subjects[s],
            np.ones((1, len(pos)), dtype=np.float32),
            positions=pos,
            n_brain=n_brain,
        )
        inside_roi[s] = roi[0] > 0

    source, small_shape = downsample_indices(shape, resolution=resolution)
    shared = np.logical_and.reduce([inside_roi[s][source] for s in datasets])
    keep = source[shared]
    xyz = np.stack(np.unravel_index(np.flatnonzero(shared), small_shape), axis=1)

    converted = {}
    for s, datasets_ in datasets.items():
        pos, n_brain = positions[s]
        converted[s] = {}
        for repetition, dataset in datasets_.items():
            values = np.nan_to_num(dataset.values, nan=0).astype(np.float32)
            mni = np.empty((len(values), len(keep)), dtype=np.float32)
            for start in tqdm(
                range(0, len(values), batch_size),
                desc=f"{subject_ids[s]} repetition {repetition}",
            ):
                batch = values[start : start + batch_size]
                volumes, _ = to_mni305(subjects[s], batch, positions=pos, n_brain=n_brain)
                mni[start : start + len(batch)] = volumes[:, keep]

            coords = {
                str(coord): ("presentation", dataset[coord].values)
                for coord in dataset.coords
                if dataset[coord].dims == ("presentation",) and coord != "presentation"
            }
            coords |= {
                "x": ("neuroid", xyz[:, 0]),
                "y": ("neuroid", xyz[:, 1]),
                "z": ("neuroid", xyz[:, 2]),
            }
            converted[s][repetition] = (
                xr.DataArray(
                    data=mni,
                    dims=("presentation", "neuroid"),
                    coords=coords,
                    name=f"{dataset.name}.mni305.to_template.resampled.{resolution}mm",
                    attrs=dataset.attrs,
                )
                .set_xindex(["stimulus", "repetition"])
                .set_xindex(["x", "y", "z"])
            )
    return converted
