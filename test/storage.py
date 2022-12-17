import multiprocessing
import os
import random
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Any

import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from humanfriendly import parse_size, parse_timespan
from plotly.subplots import make_subplots  # type: ignore

pio.kaleido.scope.mathjax = None

RESULTS_DIR_PATH: Path = (
    (Path(__file__).absolute().parent.parent).joinpath("results").joinpath("Storage")
)


def start_test(args: Namespace, config: ConfigParser) -> None:
    if args.extension == "base":
        from src.storage_base_extension import Backup, Node, get_lost_blocks

    elif args.extension == "advanced":
        from src.storage_advanced_extension import (Backup, Node,
                                                    get_lost_blocks)

    else:
        from src.storage import Backup, Node, get_lost_blocks

    print(
        f"\nStarting 100 simulations with:\n{args} - {config.get('peer', 'average_lifetime')}\n",
        flush=True,
    )

    lost_blocks_arr: list[int] = []
    safe_sims: int = 0

    for _ in range(100):
        nodes: list[Node] = []  # we build the list of nodes to pass to the Backup class

        for node_class in config.sections():
            class_config: SectionProxy = config[node_class]
            # list comprehension: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
            cfg: list[str | int | float] = [
                parse(class_config[name]) for name, parse in parsing_functions
            ]
            # the `callable(p1, p2, *args)` idiom is equivalent to `callable(p1, p2, args[0], args[1], ...)
            nodes.extend(
                Node(f"{node_class}-{i}", *cfg)  # type: ignore
                for i in range(class_config.getint("number"))
            )

        sim = Backup(nodes)  # type: ignore
        sim.run(parse_timespan(args.max_t))

        lost_blocks: int = get_lost_blocks(sim.nodes)  # type: ignore
        lost_blocks_arr.append(lost_blocks)

        if lost_blocks == 0:
            safe_sims += 1

    print(
        f"\nResults for simulation with:\n{args} - {config.get('peer', 'average_lifetime')}\n",
        flush=True,
    )
    print(f"Safe Simulations: {safe_sims}")
    print(f"Lost Blocks Average: {sum(lost_blocks_arr) / len(lost_blocks_arr)}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("config", help="configuration file")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--max-t", default="2 years")
    args: Namespace = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable

    # functions to parse every parameter of peer configuration
    parsing_functions: list[tuple[str, Any]] = [
        ("n", int),
        ("k", int),
        ("data_size", parse_size),
        ("storage_size", parse_size),
        ("upload_speed", parse_size),
        ("download_speed", parse_size),
        ("average_uptime", parse_timespan),
        ("average_downtime", parse_timespan),
        ("average_lifetime", parse_timespan),
        ("average_recover_time", parse_timespan),
        ("arrival_time", parse_timespan),
    ]
    config: ConfigParser = ConfigParser()
    config.read(args.config)

    processes: list[multiprocessing.Process] = []
    try:
        for ext in ["", "base", "advanced"]:
            args.extension = ext

            for lifetime in [
                "8 days",
                "16 days",
                "32 days",
                "64 days",
                "128 days",
                "256 days",
                "512 days",
            ]:
                config.set("peer", "average_lifetime", lifetime)
                if multiprocessing:
                    process = multiprocessing.Process(
                        target=start_test, args=(args, config)
                    )
                    processes.append(process)
                    process.start()
                else:
                    start_test(args, config)
        for process in processes:
            process.join()
    except KeyboardInterrupt as e:
        for process in processes:
            process.kill()
        raise e
