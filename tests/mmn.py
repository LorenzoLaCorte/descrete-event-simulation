import math
import multiprocessing
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from copy import deepcopy
from pathlib import Path

import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from src import MMN

pio.kaleido.scope.mathjax = None


RESULTS_DIR_PATH: Path = (
    (Path(__file__).absolute().parent.parent).joinpath("results").joinpath("MMN")
)


def start_simulation(
    fig: go.Figure, args: Namespace, color: str = "#003f5c", row: int = 1, col: int = 1
) -> None:
    print(f"\nStarting simulation with:\n{args}\n", flush=True)

    sim: MMN = MMN(
        args.lambd, args.mu, args.n, args.random, args.queues_sample, args.max_t
    )
    sim.run(args.max_t)
    completions: dict[int, float] = sim.completions
    W: float = (
        sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)
    ) / len(completions)

    print(
        f"Results for simulation with:\n{args}\nAverage time spent in the system: {W}\n"
        f"Theoretical expectation for random server choice: {1 / (args.mu - args.lambd)}",
        flush=True,
    )

    if not args.random:
        acc: int = 0
        total: int = sum(sim.stats.queues_lengths.values())
        normalized_queues_lengths: dict[int, float] = {}

        for length, occurrences in sorted(sim.stats.queues_lengths.items()):
            normalized_queues_lengths[length] = (total - acc) / total
            acc += occurrences

        fig.append_trace(  # type: ignore
            go.Scatter(
                x=list(normalized_queues_lengths.keys()),
                y=list(normalized_queues_lengths.values()),
                line_color=color,
                line_width=3,
                name=str(args.lambd),
                mode="lines",
                showlegend=(row == 1 and col == 1),
                legendgroup=1,
            ),
            row=row,
            col=col,
        )


def start_test(args: Namespace) -> None:  # noqa: C901
    queues_sample_dict: dict[float, tuple[int, int]] = {
        1 / args.n: (1, 1),
        2 / args.n: (1, 2),
        5 / args.n: (2, 1),
        10 / args.n: (2, 2),
    }
    fig: go.Figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{math.ceil(args.n * queues_sample)} choice/s"
            for queues_sample in queues_sample_dict.keys()
        ],
    )
    processes: list[multiprocessing.Process] = []
    try:
        for queues_sample, pos in queues_sample_dict.items():
            row: int
            col: int
            row, col = pos
            args.queues_sample = queues_sample
            for lambd, color in {
                0.5: "#58508d",
                0.9: "#bc5090",
                0.95: "#ff6361",
                0.99: "#ffa600",
            }.items():
                args.lambd = lambd
                if args.multiprocessing_level > 1:
                    process: multiprocessing.Process = multiprocessing.Process(
                        target=start_simulation,
                        args=(
                            fig,
                            deepcopy(args),
                            color,
                            row,
                            col,
                        ),
                    )
                    processes.append(process)
                else:
                    start_simulation(fig, args, color, row, col)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        fig.update_layout(  # type: ignore
            {
                "autosize": False,
                "height": 720,
                "width": 1080,
                "legend_title_text": "λ",
                "xaxis_title": "queue length",
                "yaxis_title": "fraction of queues with at least that size",
            }
        )
        fig.update_xaxes({"range": [0, 14], "tick0": 0, "dtick": 2})  # type: ignore
        fig.update_yaxes({"range": [0, 1], "tick0": 0, "dtick": 0.2})  # type: ignore
        fig.write_image(RESULTS_DIR_PATH.joinpath(f"n{args.n}mu{args.mu}.pdf"))  # type: ignore
    except KeyboardInterrupt as e:
        for process in processes:
            process.kill()
        raise e


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--multiprocessing-level", type=int, default=1)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max-t", type=float, default=1_000_000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--csv", help="CSV file in which to store results")
    parser.add_argument("--random", default=False, action=BooleanOptionalAction)
    parser.add_argument("--queues-sample", type=float, default=0.1)
    args: Namespace = parser.parse_args()

    for n in [10, 100, 1000]:
        args.n = n
        args.max_t = 10_000
        for mu in [1]:
            args.mu = mu
            start_test(args)
