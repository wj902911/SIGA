#!/usr/bin/env python3
"""Plot analytical electric potential from follower-moment result output."""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "build"
    / "example"
    / "Release"
    / "flexoelectricity_2DBeamBending_followerMoment"
)
DEFAULT_CASE_REGEX = r"output_l0(?:_|$)"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class RunMetadata:
    beam_length: float
    height: float
    num_elements_l: int
    num_elements_h: int
    youngs_modulus: float
    poissons_ratio: float
    length_scale: float
    dielectric_permittivity: float
    mu_l: float
    mu_t: float
    mu_s: float


@dataclass(frozen=True)
class CurvatureResult:
    timestep: float
    vts_path: Path
    finite_difference: float
    circle_fit: float


@dataclass(frozen=True)
class PotentialResult:
    y: np.ndarray
    phi: np.ndarray
    electric_field_2: np.ndarray
    analytical_model: str
    lambda_e: float
    mu_e: float
    cl: float
    ct: float
    mu_eff: float
    h_l_star: float
    h_t_star: float
    boundary_layer_length: float | None
    eta: float
    a0: float
    b0: float
    voltage_reference: float
    vacuum_permittivity: float
    is_local_limit: bool


@dataclass(frozen=True)
class VtsGrid:
    nx: int
    ny: int
    nz: int
    reference: np.ndarray
    deformed: np.ndarray
    electric_potential: np.ndarray | None = None


@dataclass(frozen=True)
class MidspanPotentialComparison:
    y_numerical: np.ndarray
    phi_numerical: np.ndarray
    y_analytical: np.ndarray
    phi_analytical: np.ndarray
    x_reference: float
    x_index: int


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_float(pattern: str, text: str, label: str, path: Path) -> float:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot find {label} in {path}")
    return float(match.group(1))


def parse_log(log_path: Path) -> RunMetadata:
    text = read_text(log_path)

    beam_length = first_float(
        rf"\bBeam\s+length\s*:\s*({FLOAT_PATTERN})",
        text,
        "beam length",
        log_path,
    )
    height = first_float(
        rf"\bBeam\s+height\s*:\s*({FLOAT_PATTERN})",
        text,
        "beam height",
        log_path,
    )
    num_elements = re.search(
        r"\bNumber\s+of\s+elements\s*:\s*(\d+)\s*x\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if not num_elements:
        raise ValueError(f"Cannot find 'Number of elements: <L> x <H>' in {log_path}")

    mechanical = re.search(
        rf"Mechanical\s+material\s*:\s*Y\s*=\s*({FLOAT_PATTERN})\s*,\s*"
        rf"nu\s*=\s*({FLOAT_PATTERN})\s*,\s*length\s+scale\s*=\s*({FLOAT_PATTERN})",
        text,
        re.IGNORECASE,
    )
    if not mechanical:
        raise ValueError(
            "Cannot find 'Mechanical material: Y = ..., nu = ..., length scale = ...' "
            f"in {log_path}"
        )

    dielectric = first_float(
        rf"\bDielectric\s+permittivity\s*:\s*({FLOAT_PATTERN})",
        text,
        "dielectric permittivity",
        log_path,
    )

    flexo = re.search(
        rf"Flexoelectric\s+tensor\s*:\s*mu_L\s*=\s*({FLOAT_PATTERN})\s*,\s*"
        rf"mu_T\s*=\s*({FLOAT_PATTERN})\s*,\s*mu_S\s*=\s*({FLOAT_PATTERN})",
        text,
        re.IGNORECASE,
    )
    if not flexo:
        raise ValueError(
            "Cannot find 'Flexoelectric tensor: mu_L = ..., mu_T = ..., mu_S = ...' "
            f"in {log_path}"
        )

    return RunMetadata(
        beam_length=beam_length,
        height=height,
        num_elements_l=int(num_elements.group(1)),
        num_elements_h=int(num_elements.group(2)),
        youngs_modulus=float(mechanical.group(1)),
        poissons_ratio=float(mechanical.group(2)),
        length_scale=float(mechanical.group(3)),
        dielectric_permittivity=dielectric,
        mu_l=float(flexo.group(1)),
        mu_t=float(flexo.group(2)),
        mu_s=float(flexo.group(3)),
    )


def resolve_existing_path(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path.resolve()
    if path.exists():
        return path.resolve()

    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative.resolve()

    raise FileNotFoundError(f"Path does not exist: {path}")


def pvd_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.pvd") if path.is_file())


def is_case_folder(folder: Path) -> bool:
    return (folder / "log.txt").is_file() and bool(pvd_files(folder))


def find_case_folder(input_path: Path, case_regex: str | None) -> Path:
    root = resolve_existing_path(input_path)
    if root.is_file():
        root = root.parent

    if is_case_folder(root):
        return root

    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a result folder: {root}")

    candidates = [
        log_path.parent
        for log_path in root.rglob("log.txt")
        if is_case_folder(log_path.parent)
    ]
    if not candidates:
        raise FileNotFoundError(f"No case folder with both log.txt and a .pvd found under {root}")

    selected = candidates
    if case_regex:
        pattern = re.compile(case_regex, re.IGNORECASE)
        filtered = [folder for folder in candidates if pattern.search(str(folder))]
        if filtered:
            selected = filtered
        else:
            print(
                f"Warning: no case matched --case-regex {case_regex!r}; "
                "using the newest case with ParaView output."
            )

    selected.sort(
        key=lambda folder: ((folder / "log.txt").stat().st_mtime, str(folder)),
        reverse=True,
    )
    return selected[0]


def find_pvd(case_folder: Path) -> Path:
    files = pvd_files(case_folder)
    if not files:
        raise FileNotFoundError(f"No .pvd file found in {case_folder}")
    if len(files) > 1:
        print(f"Warning: multiple .pvd files found; using {files[0].name}")
    return files[0]


def vts_entries_from_pvd(pvd_path: Path) -> list[tuple[float, Path]]:
    tree = ET.parse(pvd_path)
    datasets: list[tuple[float, Path]] = []
    for dataset in tree.getroot().iter("DataSet"):
        filename = dataset.attrib.get("file", "")
        if not filename.lower().endswith(".vts"):
            continue
        timestep_text = dataset.attrib.get("timestep", "nan")
        try:
            timestep = float(timestep_text)
        except ValueError:
            continue
        datasets.append((timestep, pvd_path.parent / filename))

    if not datasets:
        raise ValueError(f"No .vts datasets found in {pvd_path}")

    return sorted(datasets, key=lambda item: item[0])


def select_vts_from_pvd(pvd_path: Path, step: str) -> tuple[float, Path]:
    datasets = vts_entries_from_pvd(pvd_path)

    if step == "initial":
        timestep, vts_path = datasets[0]
    elif step == "first":
        initial_timestep = datasets[0][0]
        first_deformed = [
            item
            for item in datasets
            if item[0] > initial_timestep
        ]
        if not first_deformed:
            raise ValueError(f"No deformed step after the initial step in {pvd_path}")
        timestep, vts_path = first_deformed[0]
    elif step == "last":
        timestep, vts_path = datasets[-1]
    else:
        try:
            requested = float(step)
        except ValueError as exc:
            raise ValueError(
                "--step must be 'initial', 'first', 'last', or a numeric timestep"
            ) from exc
        matches = [
            item
            for item in datasets
            if math.isclose(item[0], requested, rel_tol=1e-12, abs_tol=1e-12)
        ]
        if not matches:
            available = ", ".join(f"{item[0]:g}" for item in datasets)
            raise ValueError(
                f"Requested timestep {requested:g} is not in {pvd_path}. "
                f"Available timesteps: {available}"
            )
        timestep, vts_path = matches[0]

    if not vts_path.is_file():
        raise FileNotFoundError(f"Selected .vts listed in {pvd_path} does not exist: {vts_path}")
    return timestep, vts_path


def parse_extent(piece: ET.Element, path: Path) -> tuple[int, int, int]:
    extent_text = piece.attrib.get("Extent")
    if extent_text is None:
        raise ValueError(f"Cannot find Piece Extent in {path}")

    values = [int(value) for value in extent_text.split()]
    if len(values) != 6:
        raise ValueError(f"Unexpected Piece Extent in {path}: {extent_text}")

    nx = values[1] - values[0] + 1
    ny = values[3] - values[2] + 1
    nz = values[5] - values[4] + 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid nonpositive Piece Extent in {path}: {extent_text}")
    return nx, ny, nz


def vtk_data_array_values(data_array: ET.Element, components: int, path: Path) -> np.ndarray:
    text = data_array.text or ""
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.size % components != 0:
        raise ValueError(
            f"DataArray in {path} has {values.size} values, not divisible by {components}"
        )
    return values.reshape((-1, components))


def named_point_data_array(root: ET.Element, name: str, path: Path) -> ET.Element:
    for data_array in root.findall(".//PointData/DataArray"):
        if data_array.attrib.get("Name") == name:
            return data_array
    raise ValueError(f"Cannot find PointData array named {name!r} in {path}")


def load_vts_grid(vts_path: Path, read_electric_potential: bool = False) -> VtsGrid:
    tree = ET.parse(vts_path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"Cannot find StructuredGrid Piece in {vts_path}")

    nx, ny, nz = parse_extent(piece, vts_path)
    expected_points = nx * ny * nz

    points_array = root.find(".//Points/DataArray")
    if points_array is None:
        raise ValueError(f"Cannot find Points DataArray in {vts_path}")
    points = vtk_data_array_values(points_array, 3, vts_path)

    displacement_array = named_point_data_array(root, "displacement", vts_path)
    displacement_components = int(displacement_array.attrib.get("NumberOfComponents", "1"))
    displacement = vtk_data_array_values(displacement_array, displacement_components, vts_path)
    if displacement_components < 2:
        raise ValueError(f"Displacement array in {vts_path} has fewer than two components")

    if points.shape[0] != expected_points or displacement.shape[0] != expected_points:
        raise ValueError(
            f"VTS point count mismatch in {vts_path}: expected {expected_points}, "
            f"got {points.shape[0]} points and {displacement.shape[0]} displacements"
        )

    deformed = points.copy()
    deformed[:, :2] += displacement[:, :2]

    electric_potential = None
    if read_electric_potential:
        potential_array = named_point_data_array(root, "electric_potential", vts_path)
        potential_components = int(potential_array.attrib.get("NumberOfComponents", "1"))
        potential = vtk_data_array_values(potential_array, potential_components, vts_path)
        if potential.shape[0] != expected_points:
            raise ValueError(
                f"Electric-potential point count mismatch in {vts_path}: "
                f"expected {expected_points}, got {potential.shape[0]}"
            )
        electric_potential = potential[:, 0].reshape((nz, ny, nx))

    return VtsGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        reference=points.reshape((nz, ny, nx, 3)),
        deformed=deformed.reshape((nz, ny, nx, 3)),
        electric_potential=electric_potential,
    )


def deformed_midline_from_vts(vts_path: Path) -> tuple[np.ndarray, np.ndarray]:
    grid = load_vts_grid(vts_path)
    z_index = grid.nz // 2
    reference_plane = grid.reference[z_index]
    deformed_plane = grid.deformed[z_index]

    target_y = 0.5 * (
        float(np.nanmin(reference_plane[:, :, 1]))
        + float(np.nanmax(reference_plane[:, :, 1]))
    )

    mid_x = np.empty(grid.nx, dtype=float)
    mid_y = np.empty(grid.nx, dtype=float)
    for ix in range(grid.nx):
        y_reference = reference_plane[:, ix, 1]
        order = np.argsort(y_reference)
        y_sorted = y_reference[order]
        x_deformed = deformed_plane[:, ix, 0][order]
        y_deformed = deformed_plane[:, ix, 1][order]
        mid_x[ix] = np.interp(target_y, y_sorted, x_deformed)
        mid_y[ix] = np.interp(target_y, y_sorted, y_deformed)

    return mid_x, mid_y


def midspan_electric_potential_comparison(
    vts_path: Path,
    metadata: RunMetadata,
    potential: PotentialResult,
) -> MidspanPotentialComparison:
    grid = load_vts_grid(vts_path, read_electric_potential=True)
    if grid.electric_potential is None:
        raise ValueError(f"Cannot find numerical electric potential in {vts_path}")

    z_index = grid.nz // 2
    reference_plane = grid.reference[z_index]
    potential_plane = grid.electric_potential[z_index]

    target_x = 0.5 * metadata.beam_length
    x_by_index = np.mean(reference_plane[:, :, 0], axis=0)
    x_index = int(np.argmin(np.abs(x_by_index - target_x)))
    x_order = np.argsort(x_by_index)
    x_sorted = x_by_index[x_order]

    y_reference = np.empty(grid.ny, dtype=float)
    phi_numerical = np.empty(grid.ny, dtype=float)
    for iy in range(grid.ny):
        y_reference[iy] = np.interp(
            target_x,
            x_sorted,
            reference_plane[iy, x_order, 1],
        )
        phi_numerical[iy] = np.interp(
            target_x,
            x_sorted,
            potential_plane[iy, x_order],
        )

    order = np.argsort(y_reference)
    y_numerical = y_reference[order] - 0.5 * metadata.height

    return MidspanPotentialComparison(
        y_numerical=y_numerical,
        phi_numerical=phi_numerical[order],
        y_analytical=potential.y,
        phi_analytical=potential.phi,
        x_reference=target_x,
        x_index=x_index,
    )


def stable_sinh_over_cosh(x: np.ndarray, denominator_arg: float) -> np.ndarray:
    denominator_arg = abs(float(denominator_arg))
    x = np.asarray(x, dtype=float)
    return (
        np.exp(x - denominator_arg) - np.exp(-x - denominator_arg)
    ) / (1.0 + math.exp(-2.0 * denominator_arg))


def stable_cosh_over_cosh(x: np.ndarray, denominator_arg: float) -> np.ndarray:
    denominator_arg = abs(float(denominator_arg))
    x = np.asarray(x, dtype=float)
    return (
        np.exp(x - denominator_arg) + np.exp(-x - denominator_arg)
    ) / (1.0 + math.exp(-2.0 * denominator_arg))


def trimmed_slice(n: int, trim_fraction: float) -> slice:
    trim = int(round(n * trim_fraction))
    if trim < 1:
        return slice(None)
    if 2 * trim >= n - 5:
        return slice(None)
    return slice(trim, n - trim)


def finite_difference_curvature(
    x_values: np.ndarray,
    y_values: np.ndarray,
    trim_fraction: float,
) -> float:
    dx = np.gradient(x_values)
    dy = np.gradient(y_values)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.power(dx * dx + dy * dy, 1.5)

    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = (dx * ddy - dy * ddx) / denominator

    interior = curvature[trimmed_slice(curvature.size, trim_fraction)]
    valid = interior[np.isfinite(interior)]
    if valid.size == 0:
        raise ValueError("Cannot compute finite-difference curvature from the deformed midline")
    return float(np.mean(valid))


def circle_fit_curvature(
    x_values: np.ndarray,
    y_values: np.ndarray,
    sign_hint: float,
    trim_fraction: float,
) -> float:
    if math.isclose(sign_hint, 0.0, rel_tol=1e-12, abs_tol=1e-14):
        return 0.0

    curve_slice = trimmed_slice(x_values.size, trim_fraction)
    x = x_values[curve_slice]
    y = y_values[curve_slice]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    matrix = np.column_stack((2.0 * x_centered, 2.0 * y_centered, np.ones_like(x_centered)))
    rhs = x_centered * x_centered + y_centered * y_centered
    cx, cy, c = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    radius_squared = c + cx * cx + cy * cy
    if radius_squared <= 0.0:
        raise ValueError("Circle fit produced a nonpositive radius")

    sign = math.copysign(1.0, sign_hint) if not math.isclose(sign_hint, 0.0) else 1.0
    return sign / math.sqrt(radius_squared)


def compute_curvature(pvd_path: Path, trim_fraction: float, step: str) -> CurvatureResult:
    timestep, vts_path = select_vts_from_pvd(pvd_path, step)
    x_values, y_values = deformed_midline_from_vts(vts_path)
    finite_difference = finite_difference_curvature(x_values, y_values, trim_fraction)
    circle_fit = circle_fit_curvature(
        x_values,
        y_values,
        finite_difference,
        trim_fraction,
    )
    return CurvatureResult(
        timestep=timestep,
        vts_path=vts_path,
        finite_difference=finite_difference,
        circle_fit=circle_fit,
    )


def analytical_potential(
    metadata: RunMetadata,
    curvature: float,
    npts: int,
    vacuum_permittivity: float,
    analytical_model: str,
) -> PotentialResult:
    nu = metadata.poissons_ratio
    young = metadata.youngs_modulus
    height = metadata.height
    length_scale = metadata.length_scale
    dielectric = metadata.dielectric_permittivity
    model = analytical_model.lower()
    if model not in {"bare", "transformed", "local"}:
        raise ValueError(
            "--analytical-model must be 'bare', 'transformed', or 'local'"
        )
    if abs(dielectric) <= 1.0e-30:
        raise ValueError("Dielectric permittivity must be nonzero")

    lambda_e = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_e = young / (2.0 * (1.0 + nu))
    cl = lambda_e + 2.0 * mu_e
    ct = lambda_e
    beta = ct / cl
    mu_eff = metadata.mu_t - metadata.mu_l * beta

    y = np.linspace(-height / 2.0, height / 2.0, npts)
    a0 = curvature * mu_eff / dielectric
    h_l = cl * length_scale * length_scale
    h_t = ct * length_scale * length_scale
    h_l_star = h_l
    h_t_star = h_t
    boundary_layer_length: float | None = None
    eta = 0.0
    b0 = 0.0
    is_local_limit = True

    if model == "bare":
        h_l_star = h_l + metadata.mu_l * metadata.mu_l / dielectric
        h_t_star = h_t + metadata.mu_l * metadata.mu_t / dielectric
    elif model == "transformed":
        if abs(dielectric - vacuum_permittivity) <= 1.0e-30:
            raise ValueError(
                "dielectricPermittivity and --vacuum-permittivity are too close "
                "for the transformed analytical solution"
            )
        h_l_star = h_l - (
            vacuum_permittivity
            * metadata.mu_l
            * metadata.mu_l
            / (dielectric * (dielectric - vacuum_permittivity))
        )
        h_t_star = h_t - (
            vacuum_permittivity
            * metadata.mu_l
            * metadata.mu_t
            / (dielectric * (dielectric - vacuum_permittivity))
        )
    else:
        h_l_star = 0.0
        h_t_star = 0.0

    coefficient_scale = max(abs(h_l_star), abs(h_t_star), 1.0)
    coefficient_tol = 1.0e-30 * coefficient_scale
    if model != "local":
        if h_l_star > coefficient_tol:
            boundary_layer_length = math.sqrt(h_l_star / cl)
            eta = h_t_star / h_l_star - beta
            b0 = metadata.mu_l * curvature * boundary_layer_length * eta / dielectric
            is_local_limit = abs(b0) <= 1.0e-30
        elif abs(h_l_star) <= coefficient_tol and abs(h_t_star) <= coefficient_tol:
            is_local_limit = True
        else:
            raise ValueError(
                "The generalized-traction analytical solution is inadmissible because "
                f"G_L = {h_l_star:.12e} is not positive."
            )

    if boundary_layer_length is None:
        electric_field_2 = np.full_like(y, a0)
        phi = -a0 * (y + height / 2.0)
    else:
        half_arg = height / (2.0 * boundary_layer_length)
        y_arg = y / boundary_layer_length
        sinh_over_half_cosh = stable_sinh_over_cosh(y_arg, half_arg)
        cosh_over_half_cosh = stable_cosh_over_cosh(y_arg, half_arg)
        phi = -a0 * (y + height / 2.0) + b0 * (
            sinh_over_half_cosh + math.tanh(half_arg)
        )
        electric_field_2 = a0 - (b0 / boundary_layer_length) * cosh_over_half_cosh

    return PotentialResult(
        y=y,
        phi=phi,
        electric_field_2=electric_field_2,
        analytical_model=model,
        lambda_e=lambda_e,
        mu_e=mu_e,
        cl=cl,
        ct=ct,
        mu_eff=mu_eff,
        h_l_star=h_l_star,
        h_t_star=h_t_star,
        boundary_layer_length=boundary_layer_length,
        eta=eta,
        a0=a0,
        b0=b0,
        voltage_reference=0.0,
        vacuum_permittivity=vacuum_permittivity,
        is_local_limit=is_local_limit,
    )


def shift_potential_reference(
    potential: PotentialResult,
    voltage_reference: float,
) -> PotentialResult:
    shift = voltage_reference - float(potential.phi[0])
    return PotentialResult(
        y=potential.y,
        phi=potential.phi + shift,
        electric_field_2=potential.electric_field_2,
        analytical_model=potential.analytical_model,
        lambda_e=potential.lambda_e,
        mu_e=potential.mu_e,
        cl=potential.cl,
        ct=potential.ct,
        mu_eff=potential.mu_eff,
        h_l_star=potential.h_l_star,
        h_t_star=potential.h_t_star,
        boundary_layer_length=potential.boundary_layer_length,
        eta=potential.eta,
        a0=potential.a0,
        b0=potential.b0,
        voltage_reference=voltage_reference,
        vacuum_permittivity=potential.vacuum_permittivity,
        is_local_limit=potential.is_local_limit,
    )


def with_analytical_potential(
    comparison: MidspanPotentialComparison,
    potential: PotentialResult,
) -> MidspanPotentialComparison:
    return MidspanPotentialComparison(
        y_numerical=comparison.y_numerical,
        phi_numerical=comparison.phi_numerical,
        y_analytical=potential.y,
        phi_analytical=potential.phi,
        x_reference=comparison.x_reference,
        x_index=comparison.x_index,
    )


def bottom_midspan_numerical_potential(
    comparison: MidspanPotentialComparison,
) -> tuple[float, float]:
    bottom_index = int(np.argmin(comparison.y_numerical))
    return (
        float(comparison.y_numerical[bottom_index]),
        float(comparison.phi_numerical[bottom_index]),
    )


def metadata_annotation(
    metadata: RunMetadata,
    curvature: float | None = None,
    x_reference: float | None = None,
    potential: PotentialResult | None = None,
) -> str:
    lines = [
        rf"$L={metadata.beam_length:.6g},\ H={metadata.height:.6g},\ \ell={metadata.length_scale:.6g}$",
        rf"$E={metadata.youngs_modulus:.6g},\ \nu={metadata.poissons_ratio:.6g}$",
        rf"$n_L={metadata.num_elements_l},\ n_H={metadata.num_elements_h}$",
    ]
    if potential is not None and abs(potential.vacuum_permittivity) > 0.0:
        lines.append(rf"$\epsilon_0={potential.vacuum_permittivity:.6g}$")
    if (
        potential is not None
        and potential.boundary_layer_length is not None
        and not potential.is_local_limit
    ):
        lines.append(
            rf"$\ell_b={potential.boundary_layer_length:.6g},\ \eta={potential.eta:.3e}$"
        )
    if curvature is not None:
        lines.append(rf"$\kappa={curvature:.6e}$")
    if x_reference is not None:
        lines.append(rf"$X={x_reference:.6g}$")
    if potential is not None and not math.isclose(
        potential.voltage_reference,
        0.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14,
    ):
        lines.append(rf"$\phi_b={potential.voltage_reference:.6e}$")
    return "\n".join(lines)


def plot_potential(
    potential: PotentialResult,
    metadata: RunMetadata,
    curvature: float,
    output_path: Path,
    thickness_scale: float,
    potential_scale: float,
    thickness_unit: str,
    potential_unit: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    ax.plot(
        potential.y * thickness_scale,
        potential.phi * potential_scale,
        color="#1f6f8b",
        linewidth=2.0,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.75)
    ax.set_xlabel(r"Thickness coordinate $Y$" + f" ({thickness_unit})")
    ax.set_ylabel(r"Electric potential $\phi$" + f" ({potential_unit})")
    ax.set_title("Analytical Electric Potential")
    ax.grid(True, color="#d8d8d8", linewidth=0.75, alpha=0.85)

    ax.text(
        0.97,
        0.03,
        metadata_annotation(metadata, curvature=curvature, potential=potential),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "alpha": 0.92,
        },
    )

    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_midspan_comparison(
    comparison: MidspanPotentialComparison,
    metadata: RunMetadata,
    potential: PotentialResult,
    output_path: Path,
    thickness_scale: float,
    potential_scale: float,
    thickness_unit: str,
    potential_unit: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    ax.plot(
        comparison.y_analytical * thickness_scale,
        comparison.phi_analytical * potential_scale,
        color="#1f6f8b",
        linewidth=2.0,
        label="Analytical",
    )
    ax.scatter(
        comparison.y_numerical * thickness_scale,
        comparison.phi_numerical * potential_scale,
        s=18,
        color="#c84630",
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
        label="Numerical",
    )
    ax.set_xlabel(r"Thickness coordinate $Y$" + f" ({thickness_unit})")
    ax.set_ylabel(r"Electric potential $\phi$" + f" ({potential_unit})")
    ax.set_title("Mid-Span Electric Potential")
    ax.grid(True, color="#d8d8d8", linewidth=0.75, alpha=0.85)
    ax.legend(frameon=True)
    ax.text(
        0.97,
        0.03,
        metadata_annotation(
            metadata,
            x_reference=comparison.x_reference,
            potential=potential,
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "alpha": 0.92,
        },
    )

    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def write_midspan_comparison_csv(
    comparison: MidspanPotentialComparison,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analytical_at_numerical_points = np.interp(
        comparison.y_numerical,
        comparison.y_analytical,
        comparison.phi_analytical,
    )
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Y",
                "numerical_electric_potential",
                "analytical_electric_potential",
                "difference_numerical_minus_analytical",
                "x_reference",
                "x_index",
            ]
        )
        for y, numerical, analytical in zip(
            comparison.y_numerical,
            comparison.phi_numerical,
            analytical_at_numerical_points,
        ):
            writer.writerow(
                [
                    f"{y:.16e}",
                    f"{numerical:.16e}",
                    f"{analytical:.16e}",
                    f"{numerical - analytical:.16e}",
                    f"{comparison.x_reference:.16e}",
                    comparison.x_index,
                ]
            )


def step_filename_suffix(step: str, timestep: float) -> str:
    if step == "last":
        return ""
    step_text = f"{timestep:g}".replace("-", "minus_").replace(".", "p")
    return f"_step_{step_text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a flexoelectric follower-moment result folder, compute the final-step "
            "beam curvature from ParaView .pvd/.vts output, read material parameters "
            "from log.txt, and plot the analytical electric potential."
        )
    )
    parser.add_argument(
        "result_path",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Case folder, or a parent folder containing case folders. Defaults to "
            "build/example/Release/flexoelectricity_2DBeamBending_followerMoment."
        ),
    )
    parser.add_argument(
        "--case-regex",
        default=DEFAULT_CASE_REGEX,
        help=(
            "Regex used only when result_path is a parent folder. "
            f"Default: {DEFAULT_CASE_REGEX!r}."
        ),
    )
    parser.add_argument(
        "--curvature-method",
        choices=["circle", "finite-difference"],
        default="circle",
        help="Curvature value to use in the analytical formula. Default: circle.",
    )
    parser.add_argument(
        "--step",
        default="last",
        help=(
            "ParaView timestep to use: 'initial' for step 0, 'first' for the first "
            "deformed/load step, 'last', or a numeric timestep. Default: last."
        ),
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.05,
        help="Fraction trimmed from each beam end before curvature averaging/fitting. Default: 0.05.",
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=501,
        help="Number of through-thickness plot samples. Default: 501.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <case_folder>/analytical_electric_potential_l0.pdf.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help=(
            "Output path for the selected-step mid-span numerical-vs-analytical plot. "
            "Defaults to <case_folder>/midspan_electric_potential_comparison_l0.pdf."
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help=(
            "CSV path for the mid-span numerical-vs-analytical data. Defaults to "
            "<case_folder>/midspan_electric_potential_comparison_l0.csv."
        ),
    )
    parser.add_argument(
        "--thickness-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to Y values for plotting. Default: 1.",
    )
    parser.add_argument(
        "--potential-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to phi values for plotting. Default: 1.",
    )
    parser.add_argument(
        "--thickness-unit",
        default="run length unit",
        help="Thickness-axis unit label. Default: 'run length unit'.",
    )
    parser.add_argument(
        "--potential-unit",
        default="run voltage unit",
        help="Potential-axis unit label. Default: 'run voltage unit'.",
    )
    parser.add_argument(
        "--vacuum-permittivity",
        type=float,
        default=0.0,
        help=(
            "Vacuum permittivity epsilon_0 used by --analytical-model transformed. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--analytical-model",
        choices=["bare", "transformed", "local"],
        default="bare",
        help=(
            "Analytical model. 'bare' uses G_L=C_L*ell^2+mu_L^2/epsilon and "
            "G_T=C_T*ell^2+mu_L*mu_T/epsilon, matching includeHbarFlexoCorrection=0. "
            "'transformed' uses the epsilon_0-corrected coefficients from the "
            "Legendre-transformed form, matching includeHbarFlexoCorrection=1. "
            "'local' uses the classical zero-boundary-layer limit. Default: bare."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the Matplotlib window after saving the figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.npts < 2:
        raise ValueError("--npts must be at least 2")
    if not 0.0 <= args.trim_fraction < 0.45:
        raise ValueError("--trim-fraction must be in [0, 0.45)")

    case_folder = find_case_folder(args.result_path, args.case_regex)
    log_path = case_folder / "log.txt"
    pvd_path = find_pvd(case_folder)

    metadata = parse_log(log_path)
    curvature_result = compute_curvature(pvd_path, args.trim_fraction, args.step)
    curvature = (
        curvature_result.circle_fit
        if args.curvature_method == "circle"
        else curvature_result.finite_difference
    )
    potential = analytical_potential(
        metadata,
        curvature,
        args.npts,
        args.vacuum_permittivity,
        args.analytical_model,
    )
    midspan_comparison_unshifted = midspan_electric_potential_comparison(
        curvature_result.vts_path,
        metadata,
        potential,
    )
    bottom_midspan_y, bottom_midspan_phi = bottom_midspan_numerical_potential(
        midspan_comparison_unshifted
    )
    potential = shift_potential_reference(potential, bottom_midspan_phi)
    midspan_comparison = with_analytical_potential(
        midspan_comparison_unshifted,
        potential,
    )

    suffix = step_filename_suffix(args.step, curvature_result.timestep)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else case_folder / f"analytical_electric_potential_l0{suffix}.pdf"
    )
    comparison_output_path = (
        args.comparison_output.resolve()
        if args.comparison_output is not None
        else case_folder / f"midspan_electric_potential_comparison_l0{suffix}.pdf"
    )
    comparison_csv_path = (
        args.comparison_csv.resolve()
        if args.comparison_csv is not None
        else case_folder / f"midspan_electric_potential_comparison_l0{suffix}.csv"
    )

    plot_potential(
        potential=potential,
        metadata=metadata,
        curvature=curvature,
        output_path=output_path,
        thickness_scale=args.thickness_scale,
        potential_scale=args.potential_scale,
        thickness_unit=args.thickness_unit,
        potential_unit=args.potential_unit,
        show=args.show,
    )
    plot_midspan_comparison(
        comparison=midspan_comparison,
        metadata=metadata,
        potential=potential,
        output_path=comparison_output_path,
        thickness_scale=args.thickness_scale,
        potential_scale=args.potential_scale,
        thickness_unit=args.thickness_unit,
        potential_unit=args.potential_unit,
        show=args.show,
    )
    write_midspan_comparison_csv(midspan_comparison, comparison_csv_path)

    print(f"Case folder: {case_folder}")
    print(f"Log file: {log_path}")
    print(
        f"Selected ParaView step: {curvature_result.timestep:g} "
        f"({curvature_result.vts_path.name})"
    )
    print(f"Curvature, finite-difference mean: {curvature_result.finite_difference:.12e}")
    print(f"Curvature, circle fit:             {curvature_result.circle_fit:.12e}")
    print(f"Curvature used:                    {curvature:.12e}")
    print(f"lambda_e = {potential.lambda_e:.12e}")
    print(f"mu_e     = {potential.mu_e:.12e}")
    print(f"analytical_model = {potential.analytical_model}")
    print(f"mu_eff_bulk = {potential.mu_eff:.12e}")
    print(f"G_L         = {potential.h_l_star:.12e}")
    print(f"G_T         = {potential.h_t_star:.12e}")
    if potential.boundary_layer_length is not None:
        print(f"ell_b       = {potential.boundary_layer_length:.12e}")
        print(f"eta         = {potential.eta:.12e}")
        print(f"A0          = {potential.a0:.12e}")
        print(f"B0          = {potential.b0:.12e}")
    print(f"E2_min      = {np.min(potential.electric_field_2):.12e}")
    print(f"E2_max      = {np.max(potential.electric_field_2):.12e}")
    print(
        "Numerical mid-span bottom potential: "
        f"{bottom_midspan_phi:.12e} at Y={bottom_midspan_y:.12e}"
    )
    print(f"Analytical voltage reference:       {potential.voltage_reference:.12e}")
    print(f"phi_bottom = {potential.phi[0]:.12e}")
    print(f"phi_top    = {potential.phi[-1]:.12e}")
    print(f"Delta_phi  = {potential.phi[-1] - potential.phi[0]:.12e}")
    print(f"Wrote {output_path}")
    print(
        f"Mid-span numerical section: x={midspan_comparison.x_reference:.12g} "
        f"(VTS x-index {midspan_comparison.x_index})"
    )
    print(f"Wrote {comparison_output_path}")
    print(f"Wrote {comparison_csv_path}")


if __name__ == "__main__":
    main()
