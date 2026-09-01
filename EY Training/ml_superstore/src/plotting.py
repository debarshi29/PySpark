"""Small helpers so every notebook saves figures the same way."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .config import FIGURE_DIR


def savefig(fig, name: str, *, dpi: int = 120) -> str:
    """Save ``fig`` as ``<FIGURE_DIR>/<name>.png`` and return the path as str."""
    fig.tight_layout()
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)


def set_style() -> None:
    """Consistent, notebook-friendly matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
