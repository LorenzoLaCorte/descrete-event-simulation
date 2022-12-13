#!/usr/bin/env python

import csv
import math
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import deque
from random import expovariate, randint, sample

from discrete_event_sim import Event, Simulation


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
        self.lambd: float = lambd
        self.n: int = n
        self.mu: float = mu
        self.arrival_rate: float = lambd
        self.completion_rate: float = mu
        self.rand: bool = rand
        self.queues_sample: float = queues_sample
        self.schedule(0, Arrival(0, self.choose_queue()))

    def choose_queue(self) -> int:
        if self.rand:
            return randint(0, len(self.queues) - 1)
        sample_queues: list[int] = sample(
            range(len(self.queues)), math.ceil(self.n * self.queues_sample)
        )
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

    if args.csv is not None:
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])
