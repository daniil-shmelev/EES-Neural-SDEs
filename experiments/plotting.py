"""Shared Matplotlib styling for paper figures."""

import matplotlib.pyplot as plt


def set_plotting_params(small_size: int, medium_size: int, bigger_size: int) -> None:
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"

    plt.rc("font", size=small_size)
    plt.rc("axes", titlesize=bigger_size)
    plt.rc("axes", labelsize=medium_size)
    plt.rc("xtick", labelsize=small_size)
    plt.rc("ytick", labelsize=small_size)
    plt.rc("legend", fontsize=small_size)
    plt.rc("figure", titlesize=bigger_size)
