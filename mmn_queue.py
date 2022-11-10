#!/usr/bin/env python

import csv
import math
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import defaultdict, deque
from random import expovariate, randint, sample

import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from discrete_event_sim import Event, Simulation

# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):
    def __init__(
        self,
        lambd: float,
        mu: float,
        n: int,
        rand: bool,
        queues_sample: float,
        max_t: float,
    ) -> None:
        super().__init__()
        self.running: list[int | None] = [
            None
        ] * n  # if not None, the id of the running job
        self.queues: list[deque[int]] = [
            deque() for _ in range(n)
        ]  # FIFO queue of the system
        self.arrivals: dict[
            int, float
        ] = {}  # dictionary mapping job id to arrival time
        self.completions: dict[
            int, float
        ] = {}  # dictionary mapping job id to completion time
        # dictionary mapping length to occurrences
        self.queues_lengths: defaultdict[int, int] = defaultdict(int)
        self.lambd: float = lambd
        self.n: int = n
        self.mu: float = mu
        self.arrival_rate: float = lambd
        self.completion_rate: float = mu
        self.rand: bool = rand
        self.queues_sample: float = queues_sample
        self.length_checker_delay: float = (max_t * 0.001) / (self.lambd * self.n)
        self.last_queues_sample_indexes: list[int] = []
        self.schedule(0, Arrival(0, self.choose_queue()))
        self.schedule(self.length_checker_delay, LengthChecker())

    def choose_queue(self) -> int:
        if self.rand:
            return randint(0, len(self.queues) - 1)
        sample_queues: list[int] = sample(
            range(len(self.queues)), math.ceil(self.n * self.queues_sample)
        )
        self.last_queues_sample_indexes = sample_queues
        return min(sample_queues, key=lambda index: self.queue_len(index))

    def schedule_arrival(self, job_id: int) -> None:
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(
            expovariate(self.lambd * self.n), Arrival(job_id, self.choose_queue())
        )

    def schedule_completion(self, job_id: int, queue_index: int) -> None:
        # schedule the time of the completion event
        self.schedule(
            expovariate(self.completion_rate), Completion(job_id, queue_index)
        )

    def queue_len(self, index: int) -> int:
        return (self.running[index] is not None) + len(self.queues[index])


class LengthChecker(Event):
    def __init__(self) -> None:
        super().__init__()

    def process(self, sim: MMN) -> None:
        for index in sim.last_queues_sample_indexes:
            sim.queues_lengths[len(sim.queues[index])] += 1
        sim.schedule(sim.length_checker_delay, LengthChecker())


class Arrival(Event):
    def __init__(self, job_id: int, queue_index: int) -> None:
        self.id: int = job_id
        self.queue_index: int = queue_index

    def process(self, sim: MMN) -> None:
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running[self.queue_index] is None:
            sim.running[self.queue_index] = self.id
            sim.schedule_completion(self.id, self.queue_index)
        # otherwise put the job into the queue
        else:
            sim.queues[self.queue_index].append(self.id)
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):
    def __init__(self, job_id: int, queue_index: int) -> None:
        self.id: int = job_id
        self.queue_index: int = queue_index

    def process(self, sim: MMN) -> None:
        assert sim.running[self.queue_index] is not None
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        # if the queue is not empty
        queue: deque[int] = sim.queues[self.queue_index]
        if queue:
            # get a job from the queue
            job: int = queue.pop()
            # schedule its completion
            sim.schedule_completion(job, self.queue_index)
        else:
            sim.running[self.queue_index] = None


def start_simulation(
    fig: go.Figure, args: Namespace, color: str = "#003f5c", row: int = 1, col: int = 1
) -> None:
    print(f"\nStarting simulation with:\n{args}\n")
    sim: MMN = MMN(
        args.lambd, args.mu, args.n, args.random, args.queues_sample, args.max_t
    )
    sim.run(args.max_t)

    completions: dict[int, float] = sim.completions
    W: float = (
        sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)
    ) / len(completions)
    print(f"Average time spent in the system: {W}")
    print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}")

    acc: int = 0
    total: int = sum(sim.queues_lengths.values())
    normalized_queues_lengths: dict[int, float] = {}

    for length, occurrences in sorted(sim.queues_lengths.items()):
        normalized_queues_lengths[length] = (total - acc) / total
        acc += occurrences

    fig.append_trace(  # type: ignore
        go.Scatter(
            x=list(normalized_queues_lengths.keys()),
            y=list(normalized_queues_lengths.values()),
            line_color=color,
            line_width=5,
            name=str(args.lambd),
            mode="lines",
            showlegend=(row == 1 and col == 1),
            legendgroup=1,
        ),
        row=row,
        col=col,
    )

    if args.csv is not None:
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max-t", type=float, default=1_000_000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--csv", help="CSV file in which to store results")
    parser.add_argument("--random", default=False, action=BooleanOptionalAction)
    parser.add_argument("--queues-sample", type=float, default=0.1)
    parser.add_argument("--test", default=False, action=BooleanOptionalAction)
    args: Namespace = parser.parse_args()

    if args.queues_sample <= 0 or args.queues_sample > 1:
        raise ValueError("queues sample must be > 0 and <= 1")

    if not args.test:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"{math.ceil(args.n * args.queues_sample)} choice/s"],
        )
        start_simulation(fig, args)
    else:
        queues_sample_dict: dict[float, tuple[int, int]] = {
            0.1: (1, 1),
            0.2: (1, 2),
            0.5: (2, 1),
            1: (2, 2),
        }
        fig: go.Figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                f"{math.ceil(args.n * queues_sample)} choice/s"
                for queues_sample in queues_sample_dict.keys()
            ],
        )
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
                start_simulation(fig, args, color, row, col)

    fig.update_layout(  # type: ignore
        {
            "legend_title_text": "λ",
            "xaxis_title": "queue length",
            "yaxis_title": "fraction of queues with at least that size",
        }
    )
    fig.update_xaxes({"range": [0, 14], "tick0": 0, "dtick": 2})  # type: ignore
    fig.update_yaxes({"range": [0, 1], "tick0": 0, "dtick": 0.2})  # type: ignore
    fig.show()  # type: ignore


if __name__ == "__main__":
    main()
