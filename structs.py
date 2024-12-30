"""Provides the types and classes used in the package."""

import random
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import Graph

GRID_SIZE: Final[int] = 5


class Coordinate(NamedTuple):
    x: int
    y: int

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def to_index(self) -> int:
        return self.y * GRID_SIZE + self.x


def coordinate_from_index(index: int) -> Coordinate:
    return Coordinate(index % GRID_SIZE, index // GRID_SIZE)


class ODPair(NamedTuple):
    origin: Coordinate
    destination: Coordinate

    def __repr__(self) -> str:
        return f"OD({self.origin} -> {self.destination})"


class User:
    def __init__(self, origin: Coordinate, destination: Coordinate):
        self.id = random.randint(0, 1000)

        self.origin = origin
        self.destination = destination

    def __repr__(self) -> str:
        return f"User {self.id} ({self.origin} -> {self.destination})"


class EnvironmentParams(TypedDict):
    bus_fares_per_km: list[float]  # Bus fares per kilometer for each bus operator
    routes_per_operator: list[int]  # Number of routes for each bus operator

    @property
    def valid(self) -> bool:
        pass


class Environment:
    def __init__(self):
        self.users: list[User] = []
        self.grid_size = GRID_SIZE
        self.grid = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        self.graph: Graph = nx.grid_2d_graph(self.grid_size, self.grid_size)

    def plot(self, *, save_to: str | Path | None = None) -> plt.Figure:
        fig = plt.figure()
        plt.gca().set_aspect("equal")

        pos = {(x, y): (x, y) for x, y in self.graph.nodes}
        nx.draw(self.graph, pos, node_size=100, with_labels=True)

        if save_to:
            fig.savefig(save_to)
        else:
            plt.show()
        return fig

    def get_random_node(self) -> Coordinate:
        """Returns a random node from the grid graph."""
        return Coordinate(*random.choice(list(self.graph.nodes)))

    def shortest_path_length(
        self, source: tuple[int, int], target: tuple[int, int]
    ) -> int:
        """Returns the shortest path length between source and target nodes in the grid graph."""
        source, target = Coordinate(*source), Coordinate(*target)
        return nx.shortest_path_length(self.graph, source, target)
