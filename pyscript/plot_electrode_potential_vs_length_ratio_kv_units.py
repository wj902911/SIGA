#!/usr/bin/env python3
"""Plot electrode electric potential against length-scale/height ratio in kV units."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FLOAT_RE = re.compile(FLOAT_PATTERN)
HEIGHT_PATTERNS = [
    re.compile(rf"\bBeam\s+height\s*[:=]\s*({FLOAT_PATTERN})", re.IGNORECASE),
    re.compile(rf"\bH\s*[:=]\s*({FLOAT_PATTERN})", re.IGNORECASE),
]
LENGTH_SCALE_PATTERNS = [
    re.compile(rf"\bLength\s+scale\s*[:=]\s*({FLOAT_PATTERN})", re.IGNORECASE),
    re.compile(rf"\blengthScale\s*[:=]\s*({FLOAT_PATTERN})"),
    re.compile(rf"\blength\s+scale\s*=\s*({FLOAT_PATTERN})", re.IGNORECASE),
]
BEAM_LENGTH_RE = re.compile(rf"\bBeam\s+length\s*[:=]\s*({FLOAT_PATTERN})", re.IGNORECASE)
NUM_ELEMENTS_RE = re.compile(r"\bNumber\s+of\s+elements\s*[:=]\s*(\d+)\s*x\s*(\d+)", re.IGNORECASE)
DEGREE_ELEVATIONS_RE = re.compile(r"\bDegree\s+elevations\s*[:=]\s*(\d+)", re.IGNORECASE)
MECHANICAL_RE = re.compile(
    rf"\bMechanical\s+material\s*:\s*Y\s*=\s*({FLOAT_PATTERN}),\s*"
    rf"nu\s*=\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
DIELECTRIC_RE = re.compile(
    rf"\bDielectric\s+permittivity\s+input\s*:\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
FLEXO_RE = re.compile(
    rf"\bFlexoelectric\s+tensor\s+input\s*:\s*mu_L\s*=\s*({FLOAT_PATTERN}),\s*"
    rf"mu_T\s*=\s*({FLOAT_PATTERN}),\s*mu_S\s*=\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
HBAR_RE = re.compile(r"\bhbar\s+flexoelectric\s+correction\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
ELECTRICAL_BC_RE = re.compile(r"\bElectrical\s+BC\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
FOLLOWER_MOMENT_RE = re.compile(
    rf"\bRight-end\s+follower\s+moment\s*[:=]\s*({FLOAT_PATTERN})",
    re.IGNORECASE,
)
MOMENT_TO_MICRONEWTON_UM = 1.0


@dataclass(frozen=True)
class RunMetadata:
    beam_length: float | None
    beam_height: float
    num_elements_l: int | None
    num_elements_h: int | None
    degree_elevations: int | None
    young_modulus: float | None
    poisson_ratio: float | None
    length_scale: float
    dielectric_permittivity: float | None
    mu_l: float | None
    mu_t: float | None
    mu_s: float | None
    hbar_correction: str | None
    electrical_bc: str | None
    follower_moment: float | None


@dataclass(frozen=True)
class PotentialPoint:
    ratio: float
    potential: float
    length_scale: float
    beam_height: float
    potential_path: Path
    log_path: Path
    metadata: RunMetadata


def parse_first_match(text: str, patterns: list[re.Pattern[str]], name: str, path: Path) -> float:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    raise ValueError(f"Cannot find {name} in {path}")


def optional_float(text: str, pattern: re.Pattern[str]) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1))


def optional_text(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def parse_run_metadata(log_path: Path) -> RunMetadata:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    beam_height = parse_first_match(text, HEIGHT_PATTERNS, "beam height", log_path)
    length_scale = parse_first_match(text, LENGTH_SCALE_PATTERNS, "length scale", log_path)
    if beam_height == 0.0:
        raise ValueError(f"Beam height is zero in {log_path}")

    elements_match = NUM_ELEMENTS_RE.search(text)
    num_elements_l = int(elements_match.group(1)) if elements_match else None
    num_elements_h = int(elements_match.group(2)) if elements_match else None

    mechanical_match = MECHANICAL_RE.search(text)
    young_modulus = float(mechanical_match.group(1)) if mechanical_match else None
    poisson_ratio = float(mechanical_match.group(2)) if mechanical_match else None

    flexo_match = FLEXO_RE.search(text)
    mu_l = float(flexo_match.group(1)) if flexo_match else None
    mu_t = float(flexo_match.group(2)) if flexo_match else None
    mu_s = float(flexo_match.group(3)) if flexo_match else None

    return RunMetadata(
        beam_length=optional_float(text, BEAM_LENGTH_RE),
        beam_height=beam_height,
        num_elements_l=num_elements_l,
        num_elements_h=num_elements_h,
        degree_elevations=(
            int(degree_match.group(1))
            if (degree_match := DEGREE_ELEVATIONS_RE.search(text))
            else None
        ),
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        length_scale=length_scale,
        dielectric_permittivity=optional_float(text, DIELECTRIC_RE),
        mu_l=mu_l,
        mu_t=mu_t,
        mu_s=mu_s,
        hbar_correction=optional_text(text, HBAR_RE),
        electrical_bc=optional_text(text, ELECTRICAL_BC_RE),
        follower_moment=optional_float(text, FOLLOWER_MOMENT_RE),
    )


def parse_potential(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = FLOAT_RE.search(text)
    if not match:
        raise ValueError(f"Cannot find electrode potential value in {path}")
    return float(match.group(0))


def nearest_log_path(potential_path: Path, root: Path) -> Path:
    root = root.resolve()
    for folder in [potential_path.parent, *potential_path.parents]:
        candidate = folder / "log.txt"
        if candidate.is_file():
            return candidate
        if folder == root:
            break
    raise FileNotFoundError(f"Cannot find log.txt above {potential_path}")


def collect_points(root: Path) -> list[PotentialPoint]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {root}")

    potential_paths = sorted(root.rglob("ElectrodeElectricPotential.txt"))
    if not potential_paths:
        raise FileNotFoundError(f"No ElectrodeElectricPotential.txt files found under {root}")

    points: list[PotentialPoint] = []
    for potential_path in potential_paths:
        log_path = nearest_log_path(potential_path, root)
        metadata = parse_run_metadata(log_path)
        potential = parse_potential(potential_path)
        points.append(
            PotentialPoint(
                ratio=metadata.length_scale / metadata.beam_height,
                potential=potential,
                length_scale=metadata.length_scale,
                beam_height=metadata.beam_height,
                potential_path=potential_path,
                log_path=log_path,
                metadata=metadata,
            )
        )
    return sorted(points, key=lambda point: (point.ratio, str(point.potential_path)))


def write_csv(points: list[PotentialPoint], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "length_scale_over_height",
                "electrode_electric_potential_kV",
                "length_scale",
                "beam_height",
                "beam_length_um",
                "young_modulus_MPa",
                "poisson_ratio",
                "dielectric_permittivity_uN_per_kV2",
                "mu_L_uN_per_kV",
                "mu_T_uN_per_kV",
                "mu_S_uN_per_kV",
                "follower_moment_microN_um",
                "potential_path",
                "log_path",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.ratio,
                    point.potential,
                    point.length_scale,
                    point.beam_height,
                    point.metadata.beam_length,
                    point.metadata.young_modulus,
                    point.metadata.poisson_ratio,
                    point.metadata.dielectric_permittivity,
                    point.metadata.mu_l,
                    point.metadata.mu_t,
                    point.metadata.mu_s,
                    point.metadata.follower_moment,
                    point.potential_path,
                    point.log_path,
                ]
            )


def companion_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def format_number(value: float) -> str:
    return f"{value:.6g}"


def format_range_tex(values: list[float]) -> str:
    if not values:
        return ""
    if math.isclose(min(values), max(values), rel_tol=1e-9, abs_tol=1e-12):
        return format_number(values[0])
    return rf"{format_number(min(values))}\ldots {format_number(max(values))}"


def format_range(values: list[float], unit: str = "") -> str:
    if not values:
        return ""
    if math.isclose(min(values), max(values), rel_tol=1e-9, abs_tol=1e-12):
        text = format_number(values[0])
    else:
        text = f"{format_number(min(values))} to {format_number(max(values))}"
    return f"{text} {unit}".rstrip()


def math_line(symbol: str, values: list[float], unit_tex: str = "") -> str:
    unit_part = rf"\,{unit_tex}" if unit_tex else ""
    return rf"${symbol} = {format_range_tex(values)}{unit_part}$"


def common_numeric_line(
    points: list[PotentialPoint],
    attr: str,
    label: str,
    unit: str = "",
) -> str | None:
    values = [getattr(point.metadata, attr) for point in points]
    if any(value is None for value in values):
        return None

    float_values = [float(value) for value in values]
    return f"{label} = {format_range(float_values, unit)}"


def common_math_line(
    points: list[PotentialPoint],
    attr: str,
    symbol: str,
    unit_tex: str = "",
    scale: float = 1.0,
) -> str | None:
    values = [getattr(point.metadata, attr) for point in points]
    if any(value is None for value in values):
        return None

    float_values = [float(value) * scale for value in values]
    return math_line(symbol, float_values, unit_tex)


def common_text_line(points: list[PotentialPoint], attr: str, label: str) -> str | None:
    values = [getattr(point.metadata, attr) for point in points]
    if any(value is None for value in values):
        return None

    first = str(values[0])
    text = first if all(str(value) == first for value in values) else "varies"
    return f"{label}: {text}"


def mesh_line(points: list[PotentialPoint]) -> str | None:
    meshes = [
        (point.metadata.num_elements_l, point.metadata.num_elements_h)
        for point in points
    ]
    if any(num_l is None or num_h is None for num_l, num_h in meshes):
        return None

    first = meshes[0]
    if all(mesh == first for mesh in meshes):
        return rf"$\mathrm{{Element\ number}} = {first[0]} \times {first[1]}$"
    return r"$\mathrm{Element\ number} = \mathrm{varies}$"


def basis_order_line(points: list[PotentialPoint]) -> str | None:
    degree_elevations = [point.metadata.degree_elevations for point in points]
    if any(value is None for value in degree_elevations):
        return None

    basis_orders = [float(value + 1) for value in degree_elevations if value is not None]
    return math_line(r"\mathrm{Basis\ order}", basis_orders)


def flexoelectric_line(points: list[PotentialPoint]) -> str | None:
    coefficient_sets = [
        (point.metadata.mu_l, point.metadata.mu_t, point.metadata.mu_s)
        for point in points
    ]
    if any(any(value is None for value in coefficient_set) for coefficient_set in coefficient_sets):
        return None

    first = coefficient_sets[0]
    if all(coefficient_set == first for coefficient_set in coefficient_sets):
        return (
            rf"$\mu_L,\mu_T,\mu_S = "
            rf"({format_number(first[0])}, {format_number(first[1])}, {format_number(first[2])})"
            r"\,\mu\mathrm{N}/\mathrm{kV}$"
        )
    return r"$\mu_L,\mu_T,\mu_S = \mathrm{varies}$"


def shorten_bc(text: str) -> str:
    replacements = {
        "bottom side grounded": "bottom grounded",
        " side equipotential electrode": " electrode",
        "remaining boundaries open circuit": "open circuit elsewhere",
    }
    shortened = text
    for old, new in replacements.items():
        shortened = shortened.replace(old, new)
    return shortened


def build_metadata_annotation(points: list[PotentialPoint]) -> str:
    lines = [r"$\bf{Simulation\ parameters}$"]

    geometry_parts = [
        common_math_line(points, "beam_length", "L", r"\mu\mathrm{m}"),
        common_math_line(points, "beam_height", "H", r"\mu\mathrm{m}"),
    ]
    geometry = ", ".join(part for part in geometry_parts if part is not None)
    if geometry:
        lines.append(geometry)

    mesh = mesh_line(points)
    basis_order = basis_order_line(points)
    if mesh:
        lines.append(mesh)
    if basis_order:
        lines.append(basis_order)

    material_parts = [
        common_math_line(points, "young_modulus", "E", r"\mathrm{MPa}"),
        common_math_line(points, "poisson_ratio", r"\nu"),
    ]
    material = ", ".join(part for part in material_parts if part is not None)
    if material:
        lines.append(material)

    dielectric = common_math_line(
        points,
        "dielectric_permittivity",
        r"\epsilon",
        r"\mu\mathrm{N}/\mathrm{kV}^2",
    )
    if dielectric:
        lines.append(dielectric)

    flexo = flexoelectric_line(points)
    if flexo:
        lines.append(flexo)

    follower_moment = common_math_line(
        points,
        "follower_moment",
        "M",
        r"\mu\mathrm{N}\,\mu\mathrm{m}",
        MOMENT_TO_MICRONEWTON_UM,
    )
    if follower_moment:
        lines.append(follower_moment)

    return "\n".join(lines)


def plot_points(
    points: list[PotentialPoint],
    output_path: Path,
    show: bool,
    log_x: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if log_x:
        plottable_points = [point for point in points if point.ratio > 0.0]
        skipped_points = len(points) - len(plottable_points)
        if skipped_points:
            print(f"Skipping {skipped_points} nonpositive length-scale ratio point(s) for log x-axis.")
        if not plottable_points:
            raise ValueError("No positive length-scale ratios available for log x-axis plotting.")
    else:
        plottable_points = points

    x_values = [point.ratio for point in plottable_points]
    y_values = [point.potential for point in plottable_points]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.14, top=0.88)
    ax.plot(x_values, y_values, color="#2f6f8f", linewidth=1.4, alpha=0.75)
    ax.scatter(x_values, y_values, color="#c84630", edgecolor="black", linewidth=0.4, zorder=3)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"Length scale ratio $\ell/H$")
    ax.set_ylabel("Electrode electric potential (kV)")
    ax.set_title(
        "Electrode Potential vs Length-Scale Ratio"
        + (" (Log X)" if log_x else " (Linear X)")
    )
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.margins(x=0.08, y=0.12)
    ax.text(
        0.98,
        0.97,
        build_metadata_annotation(points),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "alpha": 0.96,
        },
    )

    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def print_table(points: list[PotentialPoint], root: Path) -> None:
    print("length_scale/H\telectrode_potential_kV\tlength_scale\tH\tfile")
    for point in points:
        try:
            display_path = point.potential_path.relative_to(root)
        except ValueError:
            display_path = point.potential_path
        print(
            f"{point.ratio:.12g}\t"
            f"{point.potential:.12g}\t"
            f"{point.length_scale:.12g}\t"
            f"{point.beam_height:.12g}\t"
            f"{display_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find ElectrodeElectricPotential.txt files under a result folder, "
            "read length scale and beam height from the nearest log.txt, and "
            "plot electrode potential against length_scale / beam_height."
        )
    )
    parser.add_argument(
        "result_folder",
        type=Path,
        help="Result folder to search recursively.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output plot path. Defaults to "
            "<result_folder>/electrode_potential_vs_length_scale_ratio_kv_units.pdf."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for the plotted data.",
    )
    parser.add_argument(
        "--linear-output",
        type=Path,
        default=None,
        help=(
            "Output path for the linear-x plot. Defaults to the main output "
            "filename with _linear before the extension."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window after writing the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.result_folder.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "electrode_potential_vs_length_scale_ratio_kv_units.pdf"
    )
    linear_output = (
        args.linear_output.resolve()
        if args.linear_output is not None
        else companion_output_path(output, "_linear")
    )

    points = collect_points(root)
    print_table(points, root)
    plot_points(points, output, args.show, log_x=True)
    print(f"Wrote {output}")
    plot_points(points, linear_output, args.show, log_x=False)
    print(f"Wrote {linear_output}")

    if args.csv is not None:
        write_csv(points, args.csv)
        print(f"Wrote {args.csv.resolve()}")


if __name__ == "__main__":
    main()
