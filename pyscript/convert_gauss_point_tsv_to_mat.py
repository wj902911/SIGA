#!/usr/bin/env python3
"""Convert SIGA Gauss-point TSV output to a MATLAB .mat file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import savemat


STEP_RE = re.compile(r"step_(\d+)\.tsv$")


def resolve_gauss_point_folder(path: Path) -> Path:
    path = path.resolve()
    if path.name == "GaussPointData":
        return path

    candidate = path / "GaussPointData"
    if candidate.is_dir():
        return candidate.resolve()

    return path


def step_number(path: Path) -> int:
    match = STEP_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a step TSV file: {path}")
    return int(match.group(1))


def load_metadata(gauss_point_folder: Path) -> dict:
    metadata_path = gauss_point_folder / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cannot find metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tsv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find TSV file: {path}")

    data = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.float64)
    return np.atleast_2d(data)


def fill_by_element_and_gp(
    destination: np.ndarray,
    element_ids: np.ndarray,
    local_gp_ids: np.ndarray,
    values: np.ndarray,
) -> None:
    if np.any(element_ids < 0) or np.any(element_ids >= destination.shape[0]):
        raise ValueError("Element id is outside the metadata element range.")
    if np.any(local_gp_ids < 0) or np.any(local_gp_ids >= destination.shape[1]):
        raise ValueError("Gauss-point id is outside the metadata Gauss-point range.")

    destination[element_ids, local_gp_ids] = values


def convert_to_mat(
    gauss_point_folder: Path,
    output_path: Path | None,
    compress: bool,
) -> Path:
    gauss_point_folder = resolve_gauss_point_folder(gauss_point_folder)
    metadata = load_metadata(gauss_point_folder)

    num_elements = int(metadata["numElements"])
    gauss_points_per_element = int(metadata["gaussPointsPerElement"])
    num_gauss_points = int(metadata["numGaussPoints"])

    coordinates_path = gauss_point_folder / "gauss_point_parametric_coordinates.tsv"
    coordinates_table = load_tsv(coordinates_path)
    if coordinates_table.shape[0] != num_gauss_points:
        raise ValueError(
            "Coordinate TSV row count does not match metadata "
            f"({coordinates_table.shape[0]} != {num_gauss_points})."
        )
    if coordinates_table.shape[1] < 6:
        raise ValueError("Coordinate TSV must contain patch, element, gaussPoint, xi, eta, zeta.")

    coordinate_elements = coordinates_table[:, 1].astype(np.int64)
    coordinate_local_gps = coordinates_table[:, 2].astype(np.int64)
    gauss_point_parametric_coordinates = np.empty(
        (num_elements, gauss_points_per_element, 3), dtype=np.float64
    )
    fill_by_element_and_gp(
        gauss_point_parametric_coordinates,
        coordinate_elements,
        coordinate_local_gps,
        coordinates_table[:, 3:6],
    )

    step_files = sorted(gauss_point_folder.glob("step_*.tsv"), key=step_number)
    if not step_files:
        raise FileNotFoundError(f"No step_XXXX.tsv files found in {gauss_point_folder}")

    num_steps = len(step_files)
    deformation_gradient = np.empty(
        (num_steps, num_elements, gauss_points_per_element, 3, 3),
        dtype=np.float64,
    )
    strain_gradient = np.empty(
        (num_steps, num_elements, gauss_points_per_element, 3, 3, 3),
        dtype=np.float64,
    )

    for time_index, step_file in enumerate(step_files):
        print(f"Reading {step_file.name} ({time_index + 1}/{num_steps})")
        step_table = load_tsv(step_file)
        if step_table.shape[0] != num_gauss_points:
            raise ValueError(
                f"{step_file.name} row count does not match metadata "
                f"({step_table.shape[0]} != {num_gauss_points})."
            )
        if step_table.shape[1] < 39:
            raise ValueError(
                f"{step_file.name} must contain patch, element, gaussPoint, "
                "9 F columns, and 27 GradF columns."
            )

        element_ids = step_table[:, 1].astype(np.int64)
        local_gp_ids = step_table[:, 2].astype(np.int64)
        f_values = step_table[:, 3:12].reshape(num_gauss_points, 3, 3)
        grad_f_values = step_table[:, 12:39].reshape(num_gauss_points, 3, 3, 3)

        fill_by_element_and_gp(
            deformation_gradient[time_index],
            element_ids,
            local_gp_ids,
            f_values,
        )
        fill_by_element_and_gp(
            strain_gradient[time_index],
            element_ids,
            local_gp_ids,
            grad_f_values,
        )

    if output_path is None:
        output_path = gauss_point_folder / "strainGradient.mat"
    else:
        output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        output_path,
        {
            "F": deformation_gradient,
            "strainGradient": strain_gradient,
            "GaussPointParametricCoordinates": gauss_point_parametric_coordinates,
        },
        do_compression=compress,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SIGA GaussPointData TSV files to a MATLAB .mat file with "
            "F[timeStepNum, elementNum, gaussPointNum, 3, 3], "
            "strainGradient[timeStepNum, elementNum, gaussPointNum, 3, 3, 3], "
            "and GaussPointParametricCoordinates[elementNum, gaussPointNum, 3]."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to GaussPointData or to the simulation output folder containing it.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mat path. Defaults to GaussPointData/strainGradient.mat.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable scipy.io.savemat compression. Smaller file, slower write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert_to_mat(args.input, args.output, args.compress)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
