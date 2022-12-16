import logging
from collections import defaultdict
from dataclasses import dataclass
from random import expovariate

from humanfriendly import format_timespan

from .discrete_event_sim import Event, Simulation


def exp_rv(mean: float) -> float:
    """Return an exponential random variable with the given mean."""
    return expovariate(1 / mean)


def print_nodes_stats(nodes: list["Node"]) -> None:
    return


def get_safe_node_blocks(node: "Node") -> int:
    count: int = 0
    for i in range(node.n):
        if node.local_blocks[i] or node.backed_up_blocks[i]:
            count += 1
    return count


def get_lost_blocks(nodes: list["Node"]) -> int:
    lost: int = 0
    for node in nodes:
        count: int = get_safe_node_blocks(node)
        if count < node.k:
            lost += node.n
    return lost


class Backup(Simulation):
    """Backup simulation."""

    # type annotations for `Node` are strings here to allow a forward declaration:
    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    def __init__(self, nodes: list["Node"]) -> None:
        super().__init__()  # call the __init__ method of parent class
        self.nodes: list["Node"] = nodes
        # we add to the event queue the first event of each node going online and of failing
        for node in nodes:
            self.schedule(node.arrival_time, Online(node))
            self.schedule(node.arrival_time + exp_rv(node.average_lifetime), Fail(node))

    def schedule_transfer(
        self, uploader: "Node", downloader: "Node", block_id: int, restore: bool
    ) -> None:
        """Helper function called by `Node.schedule_next_upload` and `Node.schedule_next_download`.

        If `restore` is true, we are restoring a block owned by the downloader, otherwise, we are saving one owned by
        the uploader.
        """
        assert uploader.current_upload is None
        assert downloader.current_download is None

        block_size: int = downloader.block_size if restore else uploader.block_size

        speed: float = min(
            uploader.upload_speed, downloader.download_speed
        )  # we take the slowest between the two
        delay: float = block_size / speed
        event: "BlockRestoreComplete" | "BlockBackupComplete"
        if restore:
            event = BlockRestoreComplete(uploader, downloader, block_id)
        else:
            event = BlockBackupComplete(uploader, downloader, block_id)
        self.schedule(delay, event)
        uploader.current_upload = downloader.current_download = event

        self.log_info(
            f"scheduled {event.__class__.__name__} from {uploader} to {downloader}"
            f" in {format_timespan(delay)}"
        )

    def log_info(self, msg: str) -> None:
        """Override method to get human-friendly logging for time."""

        logging.info(f"{format_timespan(self.t)}: {msg}")


@dataclass(
    eq=False
)  # auto initialization from parameters below (won't consider two nodes with same state as equal)
class Node:
    """Class representing the configuration of a given node."""

    # using dataclass is (for our purposes) equivalent to having something like
    # def __init__(self, description, n, k, ...):
    #     self.n = n
    #     self.k = k
    #     ...
    #     self.__post_init__()  # if the method exists

    name: str  # the node's name

    n: int  # number of blocks in which the data is encoded
    k: int  # number of blocks sufficient to recover the whole node's data

    data_size: int  # amount of data to back up (in bytes)
    storage_size: int  # storage space devoted to storing remote data (in bytes)

    upload_speed: float  # node's upload speed, in bytes per second
    download_speed: float  # download speed

    average_uptime: float  # average time spent online
    average_downtime: float  # average time spent offline

    average_lifetime: float  # average time before a crash and data loss
    average_recover_time: float  # average time after a data loss

    arrival_time: float  # time at which the node will come online

    def __post_init__(self) -> None:
        """Compute other data dependent on config values and set up initial state."""
        self.lost = False  # ! new extension

        # whether this node is online. All nodes start offline.
        self.online: bool = False

        # whether this node is currently under repairs. All nodes are ok at start.
        self.failed: bool = False

        # size of each block
        self.block_size: int = self.data_size // self.k if self.k > 0 else 0

        # amount of free space for others' data -- note we always leave enough space for our n blocks
        self.free_space: int = self.storage_size - self.block_size * self.n

        assert self.free_space >= 0, "Node without enough space to hold its own data"

        # local_blocks[block_id] is true if we locally have the local block
        # [x] * n is a list with n references to the object x
        self.local_blocks: list[bool] = [True] * self.n

        # backed_up_blocks[block_id] is the peer we're storing that block on, or None if it's not backed up yet;
        # we start with no blocks backed up
        self.backed_up_blocks: list[list[Node]] = [[]] * self.n  # ! new extension

        # (owner -> block_id) mapping for remote blocks stored
        self.remote_blocks_held: dict[Node, list[int]] = defaultdict(
            list
        )  # ! new extension

        # current uploads and downloads, stored as a reference to the relative TransferComplete event
        self.current_upload: TransferComplete | None = None
        self.current_download: TransferComplete | None = None

    def find_block_to_back_up(self, sim: Backup) -> int | None:
        """Returns the block id of a block that needs backing up, or None if there are none."""

        # find a block that we have locally but not remotely
        # check `enumerate` and `zip`at https://docs.python.org/3/library/functions.html
        for block_id, (held_locally, peers) in enumerate(
            zip(self.local_blocks, self.backed_up_blocks, strict=False)
        ):
            if held_locally and not peers:  # ! new extension
                return block_id
        # ! new extension
        for node in sim.nodes:
            if node.online and any(not peers for peers in node.backed_up_blocks):
                return None
        # min_block_id = 0
        # min_len = len(self.backed_up_blocks[0])
        # for i, peers in enumerate(self.backed_up_blocks[1:]):
        #     curr_len = len(peers)
        #     if curr_len < min_len:
        #         min_len = curr_len
        #         min_block_id = i
        # return min_block_id
        return self.backed_up_blocks.index(sorted(self.backed_up_blocks, key=len)[0])
        # ! new extension

    def schedule_next_upload(self, sim: Backup) -> None:
        """Schedule the next upload, if any."""

        assert self.online

        sim.log_info(f"schedule_next_upload on {self}")

        if self.current_upload is not None:
            return

        # first find if we have a backup that a remote node needs
        for peer, block_ids in self.remote_blocks_held.items():
            # if the block is not present locally and the peer is online and not downloading anything currently, then
            # schedule the restore from self to peer of block_id
            for block_id in block_ids:  # ! new extension
                if (
                    peer.online
                    and not peer.local_blocks[block_id]
                    and peer.current_download is None
                ):
                    sim.schedule_transfer(
                        uploader=self, downloader=peer, block_id=block_id, restore=True
                    )
                    return  # we have found our upload, we stop

        # try to back up a block on a locally held remote node
        block_id: int | None = self.find_block_to_back_up(sim)
        if block_id is None:
            return

        sim.log_info(f"{self} is looking for somebody to back up block {block_id}")

        # ! new extension
        possible_peers: list["Node"] = sorted(
            list(set(node for nodes in self.backed_up_blocks for node in nodes)),
            key=lambda node: node.free_space,
        )
        for peer in sim.nodes:
            if peer not in possible_peers:
                possible_peers.insert(0, peer)
        # ! new extension

        for peer in possible_peers:
            # if the peer is not self, is online, is not among the remote owners, has enough space and is not
            # downloading anything currently, schedule the backup of block_id from self to peer
            if (
                peer is not self
                and peer.online
                # ! new extension
                # and peer not in remote_owners
                and peer.free_space >= peer.block_size
                and peer.current_download is None
            ):
                sim.schedule_transfer(
                    uploader=self, downloader=peer, block_id=block_id, restore=False
                )
                return

    def schedule_next_download(self, sim: Backup) -> None:
        """Schedule the next download, if any."""

        assert self.online

        sim.log_info(f"schedule_next_download on {self}")

        if self.current_download is not None:
            return

        # first find if we have a missing block to restore
        for block_id, (held_locally, peers) in enumerate(
            zip(self.local_blocks, self.backed_up_blocks, strict=False)
        ):
            for peer in peers:  # ! new extension
                if (
                    not held_locally
                    and peer is not None
                    and peer.online
                    and peer.current_upload is None
                ):
                    sim.schedule_transfer(
                        uploader=peer, downloader=self, block_id=block_id, restore=True
                    )
                    return  # we are done in this case

        # try to back up a block for a remote node
        for peer in sim.nodes:
            if (
                peer is not self
                and peer.online
                # and peer not in self.remote_blocks_held  # ! new extension
                and self.free_space >= self.block_size
                and peer.current_upload is None
            ):
                block_id: int | None = peer.find_block_to_back_up(sim)
                if block_id is not None:
                    sim.schedule_transfer(
                        uploader=peer, downloader=self, block_id=block_id, restore=False
                    )
                    return

    def __hash__(self) -> int:
        """Function that allows us to have `Node`s as dictionary keys or set items.

        With this implementation, each node is only equal to itself.
        """
        return id(self)

    def __str__(self) -> str:
        """Function that will be called when converting this to a string (e.g., when logging or printing)."""

        return self.name


@dataclass
class NodeEvent(Event):
    """An event regarding a node. Carries the identifier, i.e., the node's index in `Backup.nodes_config`"""

    node: Node

    def process(self, sim: Backup) -> None:
        """Must be implemented by subclasses."""
        raise NotImplementedError


class Online(NodeEvent):
    """A node goes online."""

    def process(self, sim: Backup) -> None:
        node: Node = self.node
        if node.online or node.failed:
            return
        node.online = True
        # schedule next upload and download
        node.schedule_next_upload(sim)
        node.schedule_next_download(sim)
        # schedule the next offline event
        sim.schedule(exp_rv(node.average_uptime), Offline(node))


class Recover(Online):
    """A node goes online after recovering from a failure."""

    def process(self, sim: Backup) -> None:
        node: Node = self.node
        sim.log_info(f"{node} recovers")
        node.failed = False
        node.free_space = node.storage_size - node.block_size * node.n
        super().process(sim)
        sim.schedule(exp_rv(node.average_lifetime), Fail(node))


class Disconnection(NodeEvent):
    """Base class for both Offline and Fail, events that make a node disconnect."""

    def process(self, sim: Backup) -> None:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def disconnect(self) -> None:
        node: Node = self.node
        node.online = False
        # cancel current upload and download
        # retrieve the nodes we're uploading and downloading to and set their current downloads and uploads to None
        current_upload: TransferComplete | None = node.current_upload
        current_download: TransferComplete | None = node.current_download
        if current_upload is not None:
            current_upload.canceled = True
            current_upload.downloader.current_download = None
            node.current_upload = None
        if current_download is not None:
            current_download.canceled = True
            current_download.uploader.current_upload = None
            node.current_download = None


class Offline(Disconnection):
    """A node goes offline."""

    def process(self, sim: Backup) -> None:
        node: Node = self.node
        if node.failed or not node.online:
            return
        assert node.online
        self.disconnect()
        # schedule the next online event
        sim.schedule(exp_rv(self.node.average_downtime), Online(node))


class Fail(Disconnection):
    """A node fails and loses all local data."""

    def process(self, sim: Backup) -> None:
        sim.log_info(f"{self.node} fails")
        self.disconnect()
        node: Node = self.node
        node.failed = True
        node.local_blocks = [False] * node.n  # lose all local data
        # ! new extension
        if not node.lost and get_safe_node_blocks(node) < node.k:
            print("Node lost")
            node.lost = True
            node.free_space = node.storage_size
            for nodes in node.backed_up_blocks:
                for remote_node in nodes:
                    remote_node.remote_blocks_held[node] = []
            node.backed_up_blocks = [[]] * node.n
        # ! new extension
        # lose all remote data
        for owner, block_ids in node.remote_blocks_held.items():
            for block_id in block_ids:  # ! new extension
                owner.backed_up_blocks[block_id].remove(node)  # ! new extension
                if owner.online and owner.current_upload is None:
                    owner.schedule_next_upload(
                        sim
                    )  # this node may want to back up the missing block
        node.remote_blocks_held.clear()
        # schedule the next online and recover events
        recover_time: float = exp_rv(node.average_recover_time)
        sim.schedule(recover_time, Recover(node))


@dataclass
class TransferComplete(Event):
    """An upload is completed."""

    uploader: Node
    downloader: Node
    block_id: int
    canceled: bool = False

    def __post_init__(self) -> None:
        assert self.uploader is not self.downloader

    def process(self, sim: Backup) -> None:
        sim.log_info(
            f"{self.__class__.__name__} from {self.uploader} to {self.downloader}"
        )
        if self.canceled:
            return  # this transfer was canceled, so ignore this event
        uploader: Node = self.uploader
        downloader: Node = self.downloader
        assert uploader.online and downloader.online
        self.update_block_state()
        uploader.current_upload = downloader.current_download = None
        uploader.schedule_next_upload(sim)
        downloader.schedule_next_download(sim)
        for node in [uploader, downloader]:
            sim.log_info(
                f"{node}: {sum(node.local_blocks)} local blocks, "
                f"{sum(peer is not None for peer in node.backed_up_blocks)} backed up blocks, "
                f"{len(node.remote_blocks_held)} remote blocks held"
            )

    def update_block_state(self) -> None:
        """Needs to be specified by the subclasses, `BackupComplete` and `DownloadComplete`."""
        raise NotImplementedError


class BlockBackupComplete(TransferComplete):
    def process(self, sim: Backup) -> None:
        return super().process(sim)

    def update_block_state(self) -> None:
        owner: Node = self.uploader
        peer: Node = self.downloader
        peer.free_space -= owner.block_size
        assert peer.free_space >= 0
        owner.backed_up_blocks[self.block_id].append(peer)  # ! new extension
        peer.remote_blocks_held[owner].append(self.block_id)  # ! new extension


class BlockRestoreComplete(TransferComplete):
    def process(self, sim: Backup) -> None:
        return super().process(sim)

    def update_block_state(self) -> None:
        owner: Node = self.downloader
        owner.local_blocks[self.block_id] = True
        if (
            sum(owner.local_blocks) == owner.k
        ):  # we have exactly k local blocks, we have all of them then
            owner.local_blocks = [True] * owner.n
