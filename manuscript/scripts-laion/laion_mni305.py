import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from laion_fmri.config import dataset_initialize
from laion_fmri.subject import load_subject

from lib.datasets.laion import DATA_DIR


MNI_RESOLUTION = 1.8
GRID_SPACING = 1.778


def _brain_indices_of_neuroids(subject, data):
    grid = np.round(subject.get_voxel_coordinates() / GRID_SPACING).astype(int)
    lookup = {tuple(voxel): i for i, voxel in enumerate(grid)}
    if len(lookup) != len(grid):
        raise ValueError(f"{subject.subject_id}: voxel grid positions are not unique")
    return np.array([
        lookup[voxel]
        for voxel in zip(data["x"].values, data["y"].values, data["z"].values)
    ])


def compute_mni305_index_map(subject, *, resolution=MNI_RESOLUTION):
    n_voxels = subject.get_n_voxels()
    image = subject.to_template(
        np.arange(1, n_voxels + 1, dtype=np.float32), "MNI305",
    )
    index_1mm = np.rint(np.asarray(image.dataobj)).astype(np.int64) - 1
    selection = [
        np.floor(np.arange(int(size // resolution)) * resolution).astype(int)
        for size in index_1mm.shape
    ]
    return index_1mm[np.ix_(*selection)]


def convert_to_mni305(
    datasets: dict[int, dict[int, xr.DataArray]],
    *,
    subjects: Sequence[str],
    resolution: float = MNI_RESOLUTION,
) -> dict[int, dict[int, xr.DataArray]]:
    dataset_initialize(DATA_DIR)

    columns = {}
    for key, splits in datasets.items():
        subject = load_subject(subjects[key])
        if not subject.has_freesurfer():
            raise FileNotFoundError(
                f"No FreeSurfer recon for {subject.subject_id}; run "
                f"download(subject='{subject.subject_id}', include_freesurfer=True)"
            )
        index = compute_mni305_index_map(subject, resolution=resolution)
        brain_to_column = np.full(subject.get_n_voxels(), -1)
        brain_to_column[_brain_indices_of_neuroids(subject, splits[0])] = np.arange(
            splits[0].sizes["neuroid"],
        )
        columns[key] = np.where(index >= 0, brain_to_column[np.maximum(index, 0)], -1)

    shapes = {c.shape for c in columns.values()}
    if len(shapes) != 1:
        raise ValueError(f"MNI305 grids differ across subjects: {shapes}")

    shared = np.logical_and.reduce([c >= 0 for c in columns.values()])
    ijk = np.argwhere(shared)
    if len(ijk) == 0:
        raise ValueError("No MNI305 voxels are shared by all subjects")
    hash_ = hashlib.blake2b(ijk.astype(np.int32).tobytes(), digest_size=4).hexdigest()

    return {
        key: {
            repetition: (
                data.isel(neuroid=columns[key][shared])
                .drop_vars(["neuroid", "x", "y", "z"], errors="ignore")
                .assign_coords(
                    x=("neuroid", ijk[:, 0]),
                    y=("neuroid", ijk[:, 1]),
                    z=("neuroid", ijk[:, 2]),
                )
                .set_xindex(["x", "y", "z"])
                .rename(f"{data.name}.mni305.{resolution}mm.voxels={hash_}")
            )
            for repetition, data in splits.items()
        }
        for key, splits in datasets.items()
    }


def save_mni305(
    datasets_mni: dict[int, dict[int, xr.DataArray]],
    *,
    subjects: Sequence[str],
    directory: str | Path,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, splits in datasets_mni.items():
        for repetition, data in splits.items():
            (
                data.reset_index(["presentation", "neuroid"])
                .rename("betas")
                .assign_attrs(name=str(data.name), subjects=",".join(subjects))
                .to_netcdf(directory / f"{subjects[key]}_rep-{repetition}.nc")
            )


def load_mni305(
    *,
    subjects: Sequence[str],
    directory: str | Path,
    n_repetitions: int = 2,
) -> dict[int, dict[int, xr.DataArray]]:
    directory = Path(directory)
    datasets_mni = {}
    for key, subject in enumerate(subjects):
        datasets_mni[key] = {}
        for repetition in range(n_repetitions):
            data = xr.open_dataarray(directory / f"{subject}_rep-{repetition}.nc").load()
            if data.attrs["subjects"] != ",".join(subjects):
                raise ValueError(
                    f"{directory} was saved for subjects {data.attrs['subjects']}, "
                    f"not {','.join(subjects)}; the shared voxels differ, so convert again"
                )
            datasets_mni[key][repetition] = (
                data.set_xindex(["stimulus", "repetition"])
                .set_xindex(["x", "y", "z"])
                .rename(data.attrs["name"])
            )
    return datasets_mni
