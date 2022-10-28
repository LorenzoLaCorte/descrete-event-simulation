import logging
from heapq import heappop, heappush
from typing import Any


class Simulation:
    """Subclass this to represent the simulation state.

    Here, self.t is the simulated time and self.events is the event queue.
    """

    def __init__(self) -> None:
        """Extend this method with the needed initialization.

        You can call super().__init__() there to call the code here.
        """

        self.t = 0  # simulated time
        self.events: list[tuple[float, "Event"]] = []

    def schedule(self, delay: float, event: "Event") -> None:
        """Add an event to the event queue after the required delay."""

        heappush(self.events, (self.t + delay, event))

    def run(self, max_t: float = float("inf")) -> None:
        """Run the simulation. If max_t is specified, stop it at that time."""

        while self.events:
            t, event = heappop(self.events)
            if t > max_t:
                break
            self.t = t
            event.process(self)

    def log_info(self, msg: str) -> None:
        logging.info(f"{self.t:.2f}: {msg}")


class Event:
    """
    Subclass this to represent your events.

    You may need to define __init__ to set up all the necessary information.
    """

    def process(self, sim: Any) -> None:
        raise NotImplementedError
