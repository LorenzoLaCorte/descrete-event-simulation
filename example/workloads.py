import csv
import gzip
import math
import os
import os.path
import random
from datetime import datetime
from functools import partial
from tempfile import NamedTemporaryFile

import requests

MUSTANG_URL: str = (
    "https://ftp.pdl.cmu.edu/pub/datasets/ATLAS/mustang/mustang_release_v1.0beta.csv.gz"
)

# NOTE: if you want to shuffle a trace, have a look at the `random.shuffle` function.


def weibull_generator(shape: float, mean: float) -> partial[float]:
    """Returns a callable that outputs random variables with a Weibull distribution having the given shape and mean."""

    return partial(random.weibullvariate, mean / math.gamma(1 + 1 / shape), shape)


def isoformat2ts(date_string: str) -> float:
    return datetime.fromisoformat(date_string).timestamp()


def parse_mustang(path: str | None = None) -> list[tuple[float, float]]:
    """Parses the Mustang trace and returns a list of (delay, size) pairs."""

    if path is None:
        path = MUSTANG_URL.split("/")[-1]
    if not os.path.exists(path):
        with NamedTemporaryFile(delete=False) as tmp:
            print(
                f"Downloading Mustang dataset (temporary file: {tmp.name})...",
                end=" ",
                flush=True,
            )
            tmp.write(requests.get(MUSTANG_URL).content)
            path = tmp.name
            print("done.")
    with gzip.open(path, "rt", newline="") as f:
        result: list[tuple[float, float]] = []
        # ! removed because it is not used: last_submit: float | None = None
        for row in csv.DictReader(f):
            if row["job_status"] != "COMPLETED":
                continue
            time_columns: list[str] = ["submit_time", "start_time", "end_time"]
            try:
                submit: float
                start: float
                end: float
                submit, start, end = (
                    isoformat2ts(row[column]) for column in time_columns
                )
            except ValueError:  # some values have a missing `start_time` column. We ignore them.
                continue
            delay: float = submit  # ! removed because it is not used: - last_submit if last_submit is not None else 0
            assert delay >= 0
            result.append((delay, (end - start) * int(row["node_count"])))
    print(f"{len(result):,} jobs parsed")
    return result


def normalize_trace(
    trace: list[tuple[float, float]], lambd: float, mu: float = 1
) -> list[tuple[float, float]]:
    """Renormalize a trace such that the average delays and size are respectively `1/lambd` and `1/mu`."""

    n: int = len(trace)
    delay_sum: float = 0
    size_sum: float = 0
    for delay, size in trace:
        delay_sum += delay
        size_sum += size
    delay_factor: float = n * delay_sum / lambd
    size_factor: float = n * size_sum / mu
    return [(delay * delay_factor, size * size_factor) for delay, size in trace]


if __name__ == "__main__":  # sanity check

    normalize_trace(parse_mustang(), 0.7)

    n_items: int = 1_000_000

    for shape in 0.5, 1, 2:
        for mean in 0.5, 1, 2:
            gen: partial[float] = weibull_generator(shape, mean)
            m: float = sum(gen() for _ in range(n_items)) / n_items
            print(
                f"shape={shape:3}, mean={mean:3}; theoretical mean: {mean:.3f}; experimental mean: {m:.3f}"
            )
