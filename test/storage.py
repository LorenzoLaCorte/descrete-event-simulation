import multiprocessing
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from configparser import ConfigParser, SectionProxy
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any

import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from humanfriendly import parse_size, parse_timespan
from plotly.subplots import make_subplots  # type: ignore

pio.kaleido.scope.mathjax = None

RESULTS_DIR_PATH: Path = (
    (Path(__file__).absolute().parent.parent).joinpath("results").joinpath("storage")
)


def start_test(
    args: Namespace,
    config: ConfigParser,
    result: DictProxy,  # type: ignore
    runs: int = 1000,
) -> None:
    if args.extension == "base":
        from src.storage_base_extension import Backup, Node, get_lost_blocks

    elif args.extension == "advanced":
        from src.storage_advanced_extension import (Backup, Node,
                                                    get_lost_blocks)

    else:
        from src.storage import Backup, Node, get_lost_blocks

    print(
        f"\nStarting {runs} simulations with:\n{args} - {config.get('peer', 'average_lifetime')}\n",
        flush=True,
    )

    lost_blocks_arr: list[int] = []
    safe_sims: int = 0

    for _ in range(runs):
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

    lost_blocks_average = sum(lost_blocks_arr) / len(lost_blocks_arr)
    result[config.get("peer", "average_lifetime")] = (safe_sims, lost_blocks_average)
    print(
        f"\nResults for simulation with:\n{args} - {config.get('peer', 'average_lifetime')}\n",
        flush=True,
    )
    print(f"Safe Simulations: {safe_sims}")
    print(f"Lost Blocks Average: {lost_blocks_average}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("config", help="configuration file")
    parser.add_argument("--max-t", default="2 years")
    parser.add_argument(
        "--multiprocessing", default=False, action=BooleanOptionalAction
    )
    args: Namespace = parser.parse_args()

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

    manager = multiprocessing.Manager()
    processes: list[multiprocessing.Process] = []
    results: dict[str, dict[str, tuple[int, float]]] = {}
    try:
        for ext in ["stock", "base", "advanced"]:
            args.extension = ext
            result: DictProxy = manager.dict()  # type: ignore
            for lifetime in [
                "4 days",
                "8 days",
                "16 days",
                "32 days",
                "64 days",
                "128 days",
                "256 days",
                "512 days",
                "1024 days",
            ]:
                result[lifetime] = (0, 0)
                config.set("peer", "average_lifetime", lifetime)
                if multiprocessing:
                    process = multiprocessing.Process(
                        target=start_test, args=(args, config, result)  # type: ignore
                    )
                    processes.append(process)
                    process.start()
                else:
                    start_test(args, config, result)
            results[ext] = result  # type: ignore
        for process in processes:
            process.join()

        fig: go.Figure = make_subplots(rows=1, cols=1)
        fig2: go.Figure = make_subplots(rows=1, cols=1)

        colors: list[str] = ["#58508d", "#bc5090", "#ff6361"]
        color_index = 0
        for ext, res in results.items():
            safe_simulations: list[int] = []
            block_simulations: list[float] = []

            for el in list(res.values()):
                safe_simulations.append(el[0])
                block_simulations.append(el[1])

            fig.append_trace(  # type: ignore
                go.Scatter(
                    x=list(res.keys()),
                    y=safe_simulations,
                    line_color=colors[color_index],
                    line_width=3,
                    name=ext,
                    mode="lines+markers",
                    showlegend=True,
                    legendgroup=1,
                ),
                row=1,
                col=1,
            )
            fig2.append_trace(  # type: ignore
                go.Scatter(
                    x=list(res.keys()),
                    y=block_simulations,
                    line_color=colors[color_index],
                    line_width=3,
                    name=ext,
                    mode="lines+markers",
                    showlegend=True,
                    legendgroup=1,
                ),
                row=1,
                col=1,
            )
            color_index += 1
        fig.update_layout(  # type: ignore
            {
                "autosize": False,
                "height": 720,
                "width": 1080,
                "legend_title_text": "extension",
                "xaxis_title": "average_lifetime (1000 simulations)",
                "yaxis_title": "safe",
            }
        )
        fig2.update_layout(  # type: ignore
            {
                "autosize": False,
                "height": 720,
                "width": 1080,
                "legend_title_text": "extension",
                "xaxis_title": "average_lifetime (1000 simulations)",
                "yaxis_title": "lost blocks average",
            }
        )
        fig.update_yaxes({"range": [0, 1000], "tick0": 0, "dtick": 50})  # type: ignore
        fig2.update_yaxes({"range": [0, 100], "tick0": 0, "dtick": 5})  # type: ignore
        fig.write_image(RESULTS_DIR_PATH.joinpath("result_safeness.pdf"))  # type: ignore
        fig2.write_image(RESULTS_DIR_PATH.joinpath("result_blocks.pdf"))  # type: ignore
    except KeyboardInterrupt as e:
        for process in processes:
            process.kill()
        raise e
