import matplotlib.pyplot as plt
import csv
import sys
from typing import *


def main() -> None:
    EXPR_GROUPS: List[Tuple[str, int, int]] = [
        ("input size (N)", 8, 0),
        ("iterations", 10, 1),
        ("issue order", 2, 3),
        ("computation intensity (cycles)", 18, 2),
    ]
    DEFAULT_PARAMS: List[Union[int, str]] = [1000000, 100, "group", 32]

    lines: List[List[str]] = []
    with open(sys.argv[1]) as file:
        reader = csv.reader(file)
        for row in reader:
            lines.append(row)

    cur_line: int = 0
    for i, (param, num_lines, param_idx) in enumerate(EXPR_GROUPS):
        if param == "issue order":
            continue
        plt.figure(i)
        if param == "input size (N)" or param == "iterations":
            plt.xscale("log")
        pts: List[Tuple[float, float]] = []
        for _ in range(num_lines):
            pts.append(
                (
                    float(lines[cur_line][param_idx]),
                    float(lines[cur_line][-1]),
                )
            )
            cur_line += 1
        pts.sort(key=lambda a: a[0])
        xs: List[float] = [pt[0] for pt in pts]
        ys: List[float] = [pt[1] for pt in pts]
        plt.plot(xs, ys)
        plt.xlabel(param)
        plt.ylabel("speedup")
        unchanged_params: str = ""
        for (name, _, _), default_val in zip(EXPR_GROUPS, DEFAULT_PARAMS):
            if name != param:
                unchanged_params += f"{name} = {default_val}, "
        unchanged_params = unchanged_params.removesuffix(", ")
        plt.title(f"{param} / speedup")
        plt.figtext(
            0.5,
            0.98,
            unchanged_params,
            wrap=True,
            horizontalalignment="center",
            fontsize=8,
        )
        plt.tight_layout()
        plt.savefig(f"var_{param}.png")


if __name__ == "__main__":
    main()
