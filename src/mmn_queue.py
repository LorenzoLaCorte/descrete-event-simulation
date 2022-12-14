import math
from collections import defaultdict, deque
from random import expovariate, randint, sample

from .discrete_event_sim import Event, Simulation


class MMNStats:
    def __init__(self, sim: "MMN") -> None:
        self.queues_lengths: defaultdict[int, int] = defaultdict(int)
        self.length_checker_delay: float = (sim.max_t * 0.001) / (sim.lambd * sim.n)
        self.last_queues_sample_indexes: list[int] = []


class LengthChecker(Event):
    def __init__(self) -> None:
        super().__init__()

    def process(self, sim: "MMN") -> None:
        for index in sim.stats.last_queues_sample_indexes:
            sim.stats.queues_lengths[len(sim.queues[index])] += 1
        sim.schedule(sim.stats.length_checker_delay, LengthChecker())


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
        self.max_t: float = max_t
        self.stats: MMNStats = MMNStats(self)
        self.schedule(0, Arrival(0, self.choose_queue()))
        self.schedule(self.stats.length_checker_delay, LengthChecker())

    def choose_queue(self) -> int:
        if self.rand:
            return randint(0, len(self.queues) - 1)
        sample_queues: list[int] = sample(
            range(len(self.queues)), math.ceil(self.n * self.queues_sample)
        )
        self.stats.last_queues_sample_indexes = sample_queues
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
