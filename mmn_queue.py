#!/usr/bin/env python

import argparse
import collections
import csv
from random import expovariate

from discrete_event_sim import Event, Simulation

# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):
    def __init__(self, lambd: float, mu: float, n: int) -> None:
        if n != 1:
            raise NotImplementedError  # extend this to make it work for multiple queues

        super().__init__()
        self.running: int | None = None  # if not None, the id of the running job
        self.queue: collections.deque[
            int
        ] = collections.deque()  # FIFO queue of the system
        self.arrivals: dict[
            int, float
        ] = {}  # dictionary mapping job id to arrival time
        self.completions: dict[
            int, float
        ] = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd / n
        self.completion_rate = mu / n
        self.schedule(expovariate(lambd), Arrival(0))

    def schedule_arrival(self, job_id: int) -> None:
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id: int) -> None:
        # schedule the time of the completion event
        self.schedule(expovariate(self.completion_rate), Completion(job_id))

    @property
    def queue_len(self) -> int:
        return (self.running is None) + len(self.queue)


class Arrival(Event):
    def __init__(self, job_id: int) -> None:
        self.id = job_id

    def process(self, sim: MMN) -> None:
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running is None:
            sim.running = self.id
            sim.schedule_completion(self.id)
        # otherwise put the job into the queue
        else:
            sim.queue.append(self.id)
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):
    def __init__(self, job_id: int) -> None:
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMN) -> None:
        assert sim.running is not None
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        # if the queue is not empty
        if sim.queue:
            # get a job from the queue
            job = sim.queue.pop()
            # schedule its completion
            sim.schedule_completion(job)
        else:
            sim.running = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.7)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--max-t", type=float, default=1_000_000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--csv", help="CSV file in which to store results")
    args = parser.parse_args()

    sim = MMN(args.lambd, args.mu, args.n)
    sim.run(args.max_t)

    completions = sim.completions
    W = (
        sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)
    ) / len(completions)
    print(f"Average time spent in the system: {W}")
    print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}")

    if args.csv is not None:
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == "__main__":
    main()
