#!/usr/bin/env python3
"""Plot electrode electric potential against mesh element count."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FLOAT_RE = re.compile(FLOAT_PATTERN)
LENGTH_SCALE_PATTERNS = [
    re.compile(rf"\bLength\s+scale\s*[:=]\s*({FLOAT_PATTERN})", re.IGNORECASE),
    re.compile(rf"\blengthScale\s*[:=]\s*({FLOAT_PATTERN})"),
    re.compile(rf"\blength\s+scale\s*=\s*({FLOAT_PATTERN})", re.IGNORECASE),
]
NUM_ELEMENTS_RE = re.compile(
    r"\bNumber\s+of\s+elements\s*[:=]\s*(\d+)\s*x\s*(\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PotentialPoint:
    x_value: int
    potential: float
    length_scale: float
    num_elements_l: int
    num_elements_h: int
    total_elements: int
    potential_path: Path
    log_path: Path


def parse_num_elements(log_path: Path) -> tuple[int, int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = NUM_ELEMENTS_RE.search(text)
    if not match:
        raise ValueError(f"Cannot find 'Number of elements: <L> x <H>' in {log_path}")

    num_l = int(match.group(1))
    num_h = int(match.group(2))
    if num_l <= 0 or num_h <= 0:
        raise ValueError(f"Invalid nonpositive element count in {log_path}: {num_l} x {num_h}")
    return num_l, num_h


def parse_length_scale(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern in LENGTH_SCALE_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    raise ValueError(f"Cannot find length scale in {log_path}")


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


def select_x_value(num_l: int, num_h: int, x_component: str) -> int:
    if x_component == "L":
        return num_l
    if x_component == "H":
        return num_h
    return num_l * num_h


def collect_points(root: Path, x_component: str) -> list[PotentialPoint]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {root}")

    potential_paths = sorted(root.rglob("ElectrodeElectricPotential.txt"))
    if not potential_paths:
        raise FileNotFoundError(f"No ElectrodeElectricPotential.txt files found under {root}")

    points: list[PotentialPoint] = []
    for potential_path in potential_paths:
        log_path = nearest_log_path(potential_path, root)
        num_l, num_h = parse_num_elements(log_path)
        total = num_l * num_h
        length_scale = parse_length_scale(log_path)
        points.append(
            PotentialPoint(
                x_value=select_x_value(num_l, num_h, x_component),
                potential=parse_potential(potential_path),
                length_scale=length_scale,
                num_elements_l=num_l,
                num_elements_h=num_h,
                total_elements=total,
                potential_path=potential_path,
                log_path=log_path,
            )
        )
    return sorted(points, key=lambda point: (point.x_value, str(point.potential_path)))


def x_axis_label(x_component: str) -> str:
    if x_component == "L":
        return "Number of elements in L direction"
    if x_component == "H":
        return "Number of elements in H direction"
    return "Total number of elements"


def companion_csv_path(output_path: Path) -> Path:
    return output_path.with_suffix(".csv")


def write_csv(points: list[PotentialPoint], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_value",
                "electrode_electric_potential",
                "length_scale",
                "num_elements_L",
                "num_elements_H",
                "total_elements",
                "potential_path",
                "log_path",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.x_value,
                    point.potential,
                    point.length_scale,
                    point.num_elements_l,
                    point.num_elements_h,
                    point.total_elements,
                    point.potential_path,
                    point.log_path,
                ]
            )


def format_number(value: float) -> str:
    return f"{value:.6g}"


def build_length_scale_annotation(points: list[PotentialPoint]) -> str:
    values = [point.length_scale for point in points]
    if not values:
        return ""

    if all(math.isclose(value, values[0], rel_tol=1e-9, abs_tol=1e-12) for value in values):
        value_text = format_number(values[0])
    else:
        value_text = f"{format_number(min(values))} to {format_number(max(values))}"
    return rf"Length scale: $\ell = {value_text}$"


def plot_points(
    points: list[PotentialPoint],
    output_path: Path,
    x_component: str,
    potential_unit: str,
    log_x: bool,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [point.x_value for point in points]
    y_values = [point.potential for point in points]

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.plot(x_values, y_values, color="#2f6f8f", linewidth=1.4, alpha=0.75)
    ax.scatter(x_values, y_values, color="#c84630", edgecolor="black", linewidth=0.4, zorder=3)
    if log_x:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(x_axis_label(x_component))
    ax.set_ylabel(f"Electrode electric potential ({potential_unit})")
    ax.set_title("Electrode Potential vs Element Number")
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.margins(x=0.08, y=0.12)
    ax.text(
        0.03,
        0.96,
        build_length_scale_annotation(points),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "alpha": 0.94,
        },
    )

    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def print_table(points: list[PotentialPoint], root: Path, potential_unit: str) -> None:
    print(
        "x_value\telectrode_potential_"
        f"{potential_unit}\tlength_scale\tnumEle_L\tnumEle_H\ttotal_elements\tfile"
    )
    for point in points:
        try:
            display_path = point.potential_path.relative_to(root)
        except ValueError:
            display_path = point.potential_path
        print(
            f"{point.x_value}\t"
            f"{point.potential:.12g}\t"
            f"{point.length_scale:.12g}\t"
            f"{point.num_elements_l}\t"
            f"{point.num_elements_h}\t"
            f"{point.total_elements}\t"
            f"{display_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find ElectrodeElectricPotential.txt files under a result folder, "
            "read 'Number of elements: L x H' from each nearest log.txt, and "
            "plot electrode potential against mesh element count."
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
            "<result_folder>/electrode_potential_vs_num_elements.pdf."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to the plot path with .csv if --write-csv is used.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write the plotted data to CSV.",
    )
    parser.add_argument(
        "--x-component",
        choices=["total", "L", "H"],
        default="total",
        help="Element count to use on the x-axis. Default: total = L * H.",
    )
    parser.add_argument(
        "--potential-unit",
        default="kV",
        help="Potential unit label for the y-axis and printed table. Default: kV.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log-scaled x-axis.",
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
        else root / "electrode_potential_vs_num_elements.pdf"
    )

    points = collect_points(root, args.x_component)
    print_table(points, root, args.potential_unit)
    plot_points(points, output, args.x_component, args.potential_unit, args.log_x, args.show)
    print(f"Wrote {output}")

    csv_output = args.csv.resolve() if args.csv is not None else companion_csv_path(output)
    if args.write_csv or args.csv is not None:
        write_csv(points, csv_output)
        print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
