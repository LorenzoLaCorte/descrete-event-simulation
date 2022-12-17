import logging
import random
import sys
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser, SectionProxy
from typing import Any

from humanfriendly import parse_size, parse_timespan

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("config", help="configuration file")
    parser.add_argument("--max-t", default="100 years")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--extension", default="")
    args: Namespace = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(
            format="{levelname}:{message}", level=logging.INFO, style="{"
        )  # output info on stdout

    if args.extension == "base":
        from src.storage_base_extension import Backup, Node, get_lost_blocks

        print("Using base extension")
    elif args.extension == "advanced":
        from src.storage_advanced_extension import (Backup, Node,
                                                    get_lost_blocks)

        print("Using advanced extension")
    else:
        from src.storage import Backup, Node, get_lost_blocks

        print("Using normal version")

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

    print(f"\nStarting simulation with:\n{args}\n")
    config.write(sys.stdout)

    sim: Backup = Backup(nodes)  # type: ignore
    sim.run(parse_timespan(args.max_t))
    sim.log_info("Simulation over")

    for node in nodes:
        print(node.name)
        print(node.local_blocks)
        print([len(n) for n in node.backed_up_blocks])
        for n2 in nodes:
            print(len(node.remote_blocks_held[n2]))
            print(node in n2.remote_blocks_held and n2.remote_blocks_held[node] != [])

    for node in nodes:
        print(f"\n --- {node.name} ---")
        print(f"Local blocks: {node.local_blocks}")
        print(f"Has backup blocks: {[len(n) > 0 for n in node.backed_up_blocks]}")
        print(f"Total backup blocks: {[len(n) for n in node.backed_up_blocks]}")
        print("Blocks held by:\n")
        for other_node in nodes:
            print(f"{other_node.name}: {other_node.remote_blocks_held[node]}")

    lost_blocks: int = get_lost_blocks(sim.nodes)  # type: ignore
    print(f"Lost blocks: {lost_blocks}")

    if lost_blocks == 0:
        print("Data is safe")
    else:
        print("Data has been lost")
