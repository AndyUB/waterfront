import matplotlib.pyplot as plt
import csv
import sys
import numpy as np
from typing import *


def main() -> None:
    WARMUP: int = 5
    REPLICATION: int = 10
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

    lang: str = sys.argv[2]

    cur_group_start_line: int = WARMUP
    for i, (param, num_lines, param_idx) in enumerate(EXPR_GROUPS):
        if param == "issue order":
            cur_group_start_line += num_lines * REPLICATION
            continue

        plt.figure(i, dpi=1200)
        if param == "input size (N)" or param == "iterations":
            plt.xscale("log")

        pts: List[Tuple[int, float]] = []
        # (x, y mean, y std)
        stats: List[Tuple[int, float, float]] = []

        for j in range(num_lines):
            reps: List[float] = []
            x: int = int(lines[cur_group_start_line + REPLICATION * j][param_idx])

            for k in range(REPLICATION):
                cur_line: int = cur_group_start_line + j * REPLICATION + k
                if int(lines[cur_line][param_idx]) != x:
                    raise ValueError(
                        f"each param config should have {REPLICATION} replications"
                    )
                pts.append(
                    (
                        int(lines[cur_line][param_idx]),
                        float(lines[cur_line][-1]),
                    )
                )
                reps.append(float(lines[cur_line][-1]))

            stats.append(
                (
                    x,
                    float(np.mean(reps)),
                    float(np.std(reps)),
                )
            )

        stats.sort(key=lambda a: a[0])
        unique_xs: List[float] = [stat[0] for stat in stats]
        y_means: List[float] = [stat[1] for stat in stats]
        y_stds: List[float] = [stat[2] for stat in stats]
        plt.errorbar(
            unique_xs,
            y_means,
            yerr=y_stds,
            marker="o",
            markersize=2,
            ecolor="red",
            color="#1f77b4",
            linestyle="--",
            elinewidth=1,
        )

        xs: List[float] = [pt[0] for pt in pts]
        ys: List[float] = [pt[1] for pt in pts]
        plt.scatter(xs, ys, marker=".", s=5, color="green")

        for j, (x, y_mean, _) in enumerate(stats):

            def get_txt_align() -> Tuple[str, str, str]:
                match i:
                    case 0:
                        if lang == "py" and j == 1:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "top",
                            )
                        elif lang == "py" and j == 2:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "bottom",
                            )
                        elif lang == "py" and j == 6:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "top",
                            )
                        elif lang == "py" and j == 7:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "bottom",
                            )
                        elif j < 2:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "bottom",
                            )
                        elif j == 5:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "bottom",
                            )
                        else:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "top",
                            )
                    case 1:
                        if lang == "py" and j < 5:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "top",
                            )
                        elif lang == "py" and j < 9:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "bottom",
                            )
                        elif lang == "py" and j == 9:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "top",
                            )
                        elif j < 5:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "top",
                            )
                        else:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "bottom",
                            )
                    case 3:
                        if lang == "py" and j == 3:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "bottom",
                            )
                        elif j < 7:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "top",
                            )
                        elif j == 7:
                            return (
                                f"({x}, {y_mean:.2f})  ",
                                "right",
                                "bottom",
                            )
                        else:
                            return (
                                f"  ({x}, {y_mean:.2f})",
                                "left",
                                "bottom",
                            )
                raise ValueError(f"unexpected graph for experiment group #{i}")

            txt, ha, va = get_txt_align()

            plt.text(
                x,
                y_mean,
                txt,
                fontsize=5,
                ha=ha,
                va=va,
                color="grey",
            )

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

        cur_group_start_line += num_lines * REPLICATION


if __name__ == "__main__":
    main()
