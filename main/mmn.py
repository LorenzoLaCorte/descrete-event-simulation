import csv
import math
from argparse import ArgumentParser, BooleanOptionalAction, Namespace

from plotly import graph_objects as go  # type: ignore

from src import MMN

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max-t", type=float, default=1_000_000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--csv", help="CSV file in which to store results")
    parser.add_argument("--random", default=False, action=BooleanOptionalAction)
    parser.add_argument("--queues-sample", type=float, default=0.1)
    args: Namespace = parser.parse_args()

    if args.queues_sample <= 0 or args.queues_sample > 1:
        raise ValueError("queues sample must be > 0 and <= 1")

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

        fig: go.Figure = go.Figure(
            data=go.Scatter(
                x=list(normalized_queues_lengths.keys()),
                y=list(normalized_queues_lengths.values()),
                line_width=3,
                name=str(args.lambd),
                mode="lines",
                showlegend=True,
            )
        )
        fig.update_layout(  # type: ignore
            {
                "title": {
                    "text": "Simultation using supermarket model with "
                    f"{math.ceil(args.n * args.queues_sample)} choice/s and n={args.n} mu={args.mu} max_t={args.max_t}",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                "legend_title_text": "λ",
                "xaxis_title": "queue length",
                "yaxis_title": "fraction of queues with at least that size",
            }
        )
        fig.update_xaxes({"range": [0, 14], "tick0": 0, "dtick": 2})  # type: ignore
        fig.update_yaxes({"range": [0, 1], "tick0": 0, "dtick": 0.2})  # type: ignore
        fig.show()  # type: ignore

    if args.csv is not None:
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])
