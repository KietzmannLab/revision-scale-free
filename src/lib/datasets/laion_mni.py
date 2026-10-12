import itertools
from collections.abc import Mapping

import nibabel as nib
import numpy as np
import scipy.sparse as sp
import torch
import xarray as xr
from laion_fmri._paths import freesurfer_transforms_dir
from laion_fmri.config import dataset_initialize
from laion_fmri.io import load_nifti_mask
from laion_fmri.subject import load_subject
from nitransforms.base import ImageGrid, SpatialReference, _as_homogeneous
from nitransforms.linear import Affine
from templateflow.api import get as tflow_get
from torch.nn.functional import interpolate

from lib.datasets.laion import DATA_DIR, SUBJECTS

GRID_SPACING = 1.778


def _load_subject(subject: str):
    dataset_initialize(DATA_DIR)
    return load_subject(subject)


def load_mni305_reference() -> nib.Nifti1Image:
    path = tflow_get("MNI305", suffix="T1w", extension=".nii.gz")
    if isinstance(path, list):
        path = path[0]
    return nib.load(str(path))


def _load_functional_grid(sub) -> tuple[nib.Nifti1Image, np.ndarray]:
    mask_path = sub._brain_mask_path("anatomical")
    img = nib.load(str(mask_path))
    brain_ijk = np.argwhere(load_nifti_mask(mask_path).reshape(img.shape))
    return img, brain_ijk


def _neuroid_ijk(
    data: xr.DataArray,
    *,
    affine: np.ndarray,
    brain_ijk: np.ndarray,
) -> np.ndarray:
    ijk_h = np.hstack([brain_ijk, np.ones((len(brain_ijk), 1))])
    world = (affine @ ijk_h.T).T[:, :3]
    grid = np.round(world / GRID_SPACING).astype(int)
    lookup = {tuple(g): tuple(v) for g, v in zip(grid, brain_ijk, strict=True)}
    if len(lookup) != len(grid):
        raise ValueError("x/y/z labels are not unique on this subject's grid")
    keys = zip(data["x"].values, data["y"].values, data["z"].values, strict=True)
    return np.array([lookup[(int(x), int(y), int(z))] for x, y, z in keys])


def compute_mni_coordinates(
    sub,
    *,
    functional: nib.Nifti1Image,
    reference: nib.Nifti1Image,
) -> np.ndarray:
    lta = freesurfer_transforms_dir(sub._data_dir, sub._subject_id) / "talairach.lta"
    transform = Affine.from_filename(str(lta), fmt="fs")
    ref = SpatialReference.factory(reference)
    return ImageGrid(functional).index(
        _as_homogeneous(transform.map(ref.ndcoords), dim=ref.ndim),
    )


def interpolation_weights(
    coordinates: np.ndarray,
    *,
    neuroid_ijk: np.ndarray,
    grid_shape: tuple[int, int, int],
    order: int,
) -> sp.csr_matrix:
    n_reference = len(coordinates)
    lookup = np.full(grid_shape, -1, dtype=np.int64)
    lookup[tuple(neuroid_ijk.T)] = np.arange(len(neuroid_ijk))

    lo, hi = neuroid_ijk.min(axis=0) - 1, neuroid_ijk.max(axis=0) + 1
    candidates = np.flatnonzero(
        np.all((coordinates >= lo) & (coordinates <= hi), axis=1),
    )
    coordinates = coordinates[candidates]

    match order:
        case 0:
            corners = [(np.floor(coordinates + 0.5).astype(np.int64), None)]
        case 1:
            base = np.floor(coordinates).astype(np.int64)
            fraction = coordinates - base
            corners = [
                (
                    base + np.array(offset),
                    np.prod(np.where(offset, fraction, 1 - fraction), axis=1),
                )
                for offset in itertools.product((0, 1), repeat=3)
            ]
        case _:
            raise ValueError("order must be 0 or 1")

    rows, cols, values = [], [], []
    for index, weight in corners:
        inside = np.all((index >= 0) & (index < grid_shape), axis=1)
        neuroid = np.full(len(index), -1, dtype=np.int64)
        neuroid[inside] = lookup[tuple(index[inside].T)]
        keep = neuroid >= 0
        rows.append(candidates[keep])
        cols.append(neuroid[keep])
        values.append(np.ones(keep.sum()) if weight is None else weight[keep])

    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_reference, len(neuroid_ijk)),
    )


def build_projection(
    data: xr.DataArray,
    *,
    subject: str,
    reference: nib.Nifti1Image,
    order: int = 1,
) -> sp.csr_matrix:
    sub = _load_subject(subject)
    functional, brain_ijk = _load_functional_grid(sub)
    neuroid_ijk = _neuroid_ijk(data, affine=functional.affine, brain_ijk=brain_ijk)
    coordinates = compute_mni_coordinates(
        sub,
        functional=functional,
        reference=reference,
    )
    weights = interpolation_weights(
        coordinates,
        neuroid_ijk=neuroid_ijk,
        grid_shape=functional.shape[:3],
        order=order,
    )
    weights.eliminate_zeros()
    return weights


def downsample_indices(
    shape: tuple[int, int, int],
    *,
    resolution: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    linear = torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(1, 1, *shape)
    downsampled = interpolate(linear, scale_factor=1 / resolution).squeeze()
    return downsampled.numpy().astype(np.int64), tuple(downsampled.shape)


def _project(
    data: xr.DataArray,
    *,
    weights: sp.csr_matrix,
    xyz: np.ndarray,
    name: str,
) -> xr.DataArray:
    values = (weights @ np.nan_to_num(data.values, nan=0).T).T.astype(np.float32)
    coords = {
        str(coord): ("presentation", data[coord].values)
        for coord in data.coords
        if data[coord].dims == ("presentation",) and coord != "presentation"
    }
    coords |= {
        "x": ("neuroid", xyz[:, 0]),
        "y": ("neuroid", xyz[:, 1]),
        "z": ("neuroid", xyz[:, 2]),
    }
    return xr.DataArray(
        data=values,
        dims=("presentation", "neuroid"),
        coords=coords,
        name=name,
        attrs=data.attrs,
    )


def convert_datasets_to_mni_space(
    datasets: Mapping[int, Mapping[int, xr.DataArray]],
    /,
    *,
    subject_ids: Mapping[int, str] | None = None,
    order: int = 1,
    resolution: float = 1.8,
) -> dict[int, dict[int, xr.DataArray]]:
    if subject_ids is None:
        subject_ids = dict(enumerate(SUBJECTS))

    reference = load_mni305_reference()
    projections = {
        subject: build_projection(
            datasets_[0],
            subject=subject_ids[subject],
            reference=reference,
            order=order,
        )
        for subject, datasets_ in datasets.items()
    }

    source, shape = downsample_indices(reference.shape[:3], resolution=resolution)
    has_support = np.ones(source.size, dtype=bool)
    for weights in projections.values():
        has_support &= np.diff(weights.indptr)[source.ravel()] > 0

    shared = np.flatnonzero(has_support)
    xyz = np.stack(np.unravel_index(shared, shape), axis=1)
    rows = source.ravel()[shared]

    return {
        subject: {
            repetition: _project(
                dataset,
                weights=projections[subject][rows],
                xyz=xyz,
                name=f"{dataset.name}.mni305.order={order}.resampled.{resolution}mm",
            )
            .set_xindex(["stimulus", "repetition"])
            .set_xindex(["x", "y", "z"])
            for repetition, dataset in datasets[subject].items()
        }
        for subject in datasets
    }


def check_against_laion_to_template(
    data: xr.DataArray,
    *,
    subject: str,
    presentation: int = 0,
) -> float:
    sub = _load_subject(subject)
    functional, brain_ijk = _load_functional_grid(sub)
    neuroid_ijk = _neuroid_ijk(data, affine=functional.affine, brain_ijk=brain_ijk)

    position = np.full(functional.shape[:3], -1, dtype=np.int64)
    position[tuple(brain_ijk.T)] = np.arange(len(brain_ijk))
    full = np.zeros(len(brain_ijk), dtype=np.float32)
    values = np.nan_to_num(data.isel(presentation=presentation).values, nan=0)
    full[position[tuple(neuroid_ijk.T)]] = values

    expected = np.asarray(sub.to_template(full, "MNI305").dataobj).ravel()

    reference = load_mni305_reference()
    weights = build_projection(data, subject=subject, reference=reference, order=0)
    ours = weights @ values

    return float(np.abs(ours - expected).max())s