r"""
Figure 1: Control-bank structural-distance trajectory.

Reads the per-block control-bank scoring CSV produced by the post hoc
control-bank pass and plots structural distance against Portuguese, Sumerian,
and Markov noise as a function of block number.

The resulting figure visualizes Section 3.6 of the manuscript: Portuguese
becomes much closer than the medieval Romance validators, Sumerian closes
meaningfully but stays farther than Portuguese, and Markov never becomes
the nearest control.

Outputs:
    figures/Fig1.pdf   vector format used by pdflatex via \includegraphics
    figures/Fig1.png   300-DPI raster for separate journal upload

Run from project_rbt/ root:
    python figures/plot_control_trajectory.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank.csv")
OUT_DIR = Path("figures")

# Three controls; consistent ordering and styling
CONTROLS = [
    ("Portuguese",  "vs_portuguese_control",  "#1f77b4",  "-"),
    ("Sumerian",    "vs_sumerian",            "#ff7f0e",  "--"),
    ("Markov noise","vs_markov_noise",        "#7f7f7f",  ":"),
]


def block_number(block_id: str) -> int:
    return int(block_id.split("_")[1])


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["block_num"] = df["block_id"].map(block_number)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for label, control_id, color, linestyle in CONTROLS:
        sub = df[df["control_id"] == control_id].sort_values("block_num")
        ax.plot(
            sub["block_num"],
            sub["control_structural_distance"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
        )

    ax.set_xlabel("Block number")
    ax.set_ylabel("Structural distance to control")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "Fig1.pdf")
    fig.savefig(OUT_DIR / "Fig1.png", dpi=300)
    print(f"Wrote {OUT_DIR / 'Fig1.pdf'} and {OUT_DIR / 'Fig1.png'}")


if __name__ == "__main__":
    main()
