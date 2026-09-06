"""Compose fBm convergence panels into manuscript-ready summary PDFs.

The committed convergence outputs are one 2x2 PDF per Hurst value. This
script rearranges those files without rerunning the stochastic experiments.
By default it writes one 24-panel portrait page containing both Euclidean and
SO(3) results. It can also write the two older 12-panel pages separately.

This is intentionally a compositor rather than a rerun of the stochastic
experiments. It keeps the plotted data exactly as committed in
``experiments/convergence_fbm/results``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"

HURSTS = (0.4, 0.5, 0.6)
H_CODES = (40, 50, 60)

KINDS = {
    "euclidean": {
        "prefix": "ees_stochastic_convergence",
        "output": "ees_stochastic_convergence_12up.pdf",
    },
    "so3": {
        "prefix": "cfees_stochastic_convergence",
        "output": "cfees_stochastic_convergence_12up.pdf",
    },
}
COMBINED_OUTPUT = "convergence_fbm_24up.pdf"


def render_pdf(pdf_path: Path, out_path: Path, dpi: int) -> Image.Image:
    """Render the first page of a PDF to a transparent PNG via Ghostscript."""
    if shutil.which("gs") is None:
        raise RuntimeError("Ghostscript executable 'gs' is required to render PDFs.")

    subprocess.run(
        [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            f"-sOutputFile={out_path}",
            str(pdf_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return Image.open(out_path).convert("RGBA")


def content_box(
    image: Image.Image,
    pad: int = 18,
    threshold: int = 248,
) -> tuple[int, int, int, int]:
    """Return the non-white content box as left, upper, right, lower pixels."""
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)

    alpha = arr[..., 3]
    rgb = arr[..., :3]
    non_white = (alpha > 16) & np.any(rgb < threshold, axis=-1)
    if not np.any(non_white):
        return (0, 0, rgba.width, rgba.height)

    ys, xs = np.where(non_white)
    left = max(int(xs.min()) - pad, 0)
    upper = max(int(ys.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, rgba.width)
    lower = min(int(ys.max()) + pad + 1, rgba.height)
    return (left, upper, right, lower)


def trim_whitespace(image: Image.Image, pad: int = 18, threshold: int = 248) -> Image.Image:
    """Trim white/transparent margins while retaining a small padding."""
    rgba = image.convert("RGBA")
    left, upper, right, lower = content_box(rgba, pad=pad, threshold=threshold)
    return rgba.crop((left, upper, right, lower))


def split_2x2_page(page: Image.Image) -> list[Image.Image]:
    """Return panels in reading order: top-left, top-right, bottom-left, bottom-right."""
    width, height = page.size
    mid_x = width // 2
    mid_y = height // 2
    boxes = [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, width, mid_y),
        (0, mid_y, mid_x, height),
        (mid_x, mid_y, width, height),
    ]
    return [trim_whitespace(page.crop(box)) for box in boxes]


def load_panels(results_dir: Path, prefix: str, dpi: int) -> list[list[Image.Image]]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for code in H_CODES:
            pdf_path = results_dir / f"{prefix}_H{code:02d}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(pdf_path)

            png_path = tmp_dir / f"{pdf_path.stem}.png"
            page = render_pdf(pdf_path, png_path, dpi=dpi)
            rows.append(split_2x2_page(page))
    return rows


def panel_trims(pdf_path: Path, dpi: int) -> list[tuple[float, float, float, float]]:
    """Return graphicx trims for the four panels in left, bottom, right, top order."""
    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / f"{pdf_path.stem}.png"
        page = render_pdf(pdf_path, png_path, dpi=dpi)

        width_px, height_px = page.size
        page_w = width_px * 72.0 / dpi
        page_h = height_px * 72.0 / dpi
        mid_x = width_px // 2
        mid_y = height_px // 2
        quadrants = [
            (0, 0, mid_x, mid_y),
            (mid_x, 0, width_px, mid_y),
            (0, mid_y, mid_x, height_px),
            (mid_x, mid_y, width_px, height_px),
        ]

        trims = []
        for x0, y0, x1, y1 in quadrants:
            crop = page.crop((x0, y0, x1, y1))
            left, upper, right, lower = content_box(crop)
            global_left = x0 + left
            global_upper = y0 + upper
            global_right = x0 + right
            global_lower = y0 + lower

            trim_left = global_left * 72.0 / dpi
            trim_bottom = page_h - global_lower * 72.0 / dpi
            trim_right = page_w - global_right * 72.0 / dpi
            trim_top = global_upper * 72.0 / dpi
            trims.append((trim_left, trim_bottom, trim_right, trim_top))
    return trims


def tex_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def compose_vector_page(
    results_dir: Path,
    prefix: str,
    output_path: Path,
    *,
    dpi: int,
    page_width: float,
    page_height: float,
) -> None:
    source_paths = [results_dir / f"{prefix}_H{code:02d}.pdf" for code in H_CODES]
    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(source_path)

    trims_by_h = [panel_trims(source_path, dpi=dpi) for source_path in source_paths]
    panel_width = (page_width - 0.24) / 3
    col_gap = 0.055

    def include_panel(h_idx: int, panel_idx: int) -> str:
        trims = " ".join(f"{value:.3f}bp" for value in trims_by_h[h_idx][panel_idx])
        return (
            rf"\includegraphics[width={panel_width:.3f}in,"
            rf"trim={{{trims}}},clip]{{{tex_path(source_paths[h_idx])}}}"
        )

    header = " & ".join(rf"\footnotesize $H={hurst:.1f}$" for hurst in HURSTS)
    panel_rows = []
    for panel_idx in range(4):
        row = " & ".join(include_panel(h_idx, panel_idx) for h_idx in range(3))
        gap = r"\\[0.035in]" if panel_idx < 3 else r"\\"
        panel_rows.append(row + gap)

    tex = rf"""
\documentclass{{article}}
\usepackage[paperwidth={page_width:.3f}in,paperheight={page_height:.3f}in,margin=0in]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\null\vfill
\makebox[\paperwidth][c]{{%
\begin{{tabular}}{{@{{}}c@{{\hspace{{{col_gap:.3f}in}}}}c@{{\hspace{{{col_gap:.3f}in}}}}c@{{}}}}
{header}\\[0.020in]
{chr(10).join(panel_rows)}
\end{{tabular}}%
}}
\vfill\null
\end{{document}}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tex_path_ = tmp_dir / "figure.tex"
        tex_path_.write_text(tex)
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tmp_dir),
                str(tex_path_),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        shutil.copyfile(tmp_dir / "figure.pdf", output_path)


def compose_vector_combined_page(
    results_dir: Path,
    output_path: Path,
    *,
    dpi: int,
    page_width: float,
    page_height: float,
) -> None:
    prefixes = [KINDS["euclidean"]["prefix"], KINDS["so3"]["prefix"]]
    source_paths_by_kind = [
        [results_dir / f"{prefix}_H{code:02d}.pdf" for code in H_CODES]
        for prefix in prefixes
    ]
    for source_paths in source_paths_by_kind:
        for source_path in source_paths:
            if not source_path.exists():
                raise FileNotFoundError(source_path)

    trims_by_kind = [
        [panel_trims(source_path, dpi=dpi) for source_path in source_paths]
        for source_paths in source_paths_by_kind
    ]
    panel_width = (page_width - 0.04) / 3
    col_gap = 0.010

    def include_panel(kind_idx: int, h_idx: int, panel_idx: int) -> str:
        trims = " ".join(
            f"{value:.3f}bp" for value in trims_by_kind[kind_idx][h_idx][panel_idx]
        )
        return (
            rf"\includegraphics[width={panel_width:.3f}in,"
            rf"trim={{{trims}}},clip]{{{tex_path(source_paths_by_kind[kind_idx][h_idx])}}}"
        )

    header = " & ".join(rf"\small $H={hurst:.1f}$" for hurst in HURSTS)
    panel_rows = []
    row_specs = [(0, panel_idx) for panel_idx in range(4)] + [
        (1, panel_idx) for panel_idx in range(4)
    ]
    for row_idx, (kind_idx, panel_idx) in enumerate(row_specs):
        row = " & ".join(include_panel(kind_idx, h_idx, panel_idx) for h_idx in range(3))
        if row_idx == 3:
            gap = r"\\[0.025in]"
        elif row_idx < len(row_specs) - 1:
            gap = r"\\[0.004in]"
        else:
            gap = r"\\"
        panel_rows.append(row + gap)

    tex = rf"""
\documentclass{{article}}
\usepackage[paperwidth={page_width:.3f}in,paperheight={page_height:.3f}in,margin=0in]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\pdfhorigin=0pt
\pdfvorigin=0pt
\begin{{document}}
\newsavebox{{\convergencegrid}}
\sbox{{\convergencegrid}}{{%
\begin{{tabular}}{{@{{}}c@{{\hspace{{{col_gap:.3f}in}}}}c@{{\hspace{{{col_gap:.3f}in}}}}c@{{}}}}
{header}\\[0.015in]
{chr(10).join(panel_rows)}
\end{{tabular}}%
}}
\shipout\vbox to \paperheight{{%
\vfil
\hbox to \paperwidth{{\hfil\usebox{{\convergencegrid}}\hfil}}%
\vfil
}}
\end{{document}}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tex_path_ = tmp_dir / "figure.tex"
        tex_path_.write_text(tex)
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tmp_dir),
                str(tex_path_),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        shutil.copyfile(tmp_dir / "figure.pdf", output_path)


def compose_page(
    rows: list[list[Image.Image]],
    output_path: Path,
    *,
    dpi: int,
    page_width: float,
    page_height: float,
) -> None:
    fig = plt.figure(
        figsize=(page_width, page_height),
        dpi=dpi,
        facecolor="white",
    )

    # Portrait layout: columns are Hurst values; rows are the four panels from
    # each per-Hurst source figure. Axes positions are computed in physical
    # units so the rasterised panels are not distorted.
    panel_grid = [[rows[col_idx][row_idx] for col_idx in range(3)] for row_idx in range(4)]
    aspect = float(np.mean([panel.width / panel.height for row in panel_grid for panel in row]))

    left = 0.055
    right = 0.015
    col_gap = 0.028
    row_gap = 0.018
    header_h = 0.045

    panel_w = (1.0 - left - right - 2 * col_gap) / 3
    panel_h = panel_w * page_width / page_height / aspect
    content_h = header_h + 4 * panel_h + 3 * row_gap
    top = min(0.985, 0.5 + content_h / 2)

    for col_idx, hurst in enumerate(HURSTS):
        x0 = left + col_idx * (panel_w + col_gap)
        fig.text(
            x0 + panel_w / 2,
            top - 0.015,
            rf"$H={hurst:.1f}$",
            ha="center",
            va="top",
            fontsize=8,
        )

    first_panel_y = top - header_h - panel_h
    for row_idx, row in enumerate(panel_grid):
        y0 = first_panel_y - row_idx * (panel_h + row_gap)
        for col_idx, panel in enumerate(row):
            x0 = left + col_idx * (panel_w + col_gap)
            ax = fig.add_axes([x0, y0, panel_w, panel_h])
            ax.imshow(panel)
            ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the per-Hurst convergence PDFs.",
    )
    parser.add_argument(
        "--kind",
        choices=("combined", "all", *KINDS.keys()),
        default="combined",
        help="Which page to compose.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Rasterisation DPI used internally before assembling the page.",
    )
    parser.add_argument(
        "--page-width",
        type=float,
        default=5.5,
        help="Output PDF page width in inches.",
    )
    parser.add_argument(
        "--page-height",
        type=float,
        default=None,
        help="Output PDF page height in inches.",
    )
    parser.add_argument(
        "--raster",
        action="store_true",
        help="Force the Matplotlib raster compositor for separate 12-panel outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind in {"combined", "all"}:
        if shutil.which("pdflatex") is None:
            raise RuntimeError("The 24-panel vector compositor requires pdflatex.")
        compose_vector_combined_page(
            args.results_dir,
            args.results_dir / COMBINED_OUTPUT,
            dpi=args.dpi,
            page_width=args.page_width,
            page_height=args.page_height or 9.65,
        )
        print(f"Wrote {(args.results_dir / COMBINED_OUTPUT).resolve()}")

    kinds = KINDS.keys() if args.kind == "all" else (
        () if args.kind == "combined" else (args.kind,)
    )

    for kind in kinds:
        spec = KINDS[kind]
        output_path = args.results_dir / spec["output"]
        page_height = args.page_height or 5.8
        if not args.raster and shutil.which("pdflatex") is not None:
            compose_vector_page(
                args.results_dir,
                spec["prefix"],
                output_path,
                dpi=args.dpi,
                page_width=args.page_width,
                page_height=page_height,
            )
        else:
            rows = load_panels(args.results_dir, spec["prefix"], dpi=args.dpi)
            compose_page(
                rows,
                output_path,
                dpi=args.dpi,
                page_width=args.page_width,
                page_height=page_height,
            )
        print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
