"""Constraint-equivalent scalar A* used as a benchmark baseline.

Only the OPEN-list priority is scalarised.  Edge feasibility, altitude,
turn-radius, battery, and diversion-reserve treatment intentionally mirrors
the production EMOA* implementation.
"""

from dataclasses import dataclass
import heapq
import itertools

import numpy as np

from scripts.moa.edge_costs import NEIGHBOR_OFFSETS, edge_objectives, turn_is_feasible
from scripts.moa.heuristics import (compute_heuristics, compute_time_to_landing,
                                    compute_time_to_landing_with_turns)


@dataclass
class _Label:
    vertex: tuple[int, int]
    g: tuple[float, float, float]
    parent: object = None


class ScalarAStar:
    """A* minimizing a non-negative weighted sum of the three objectives."""

    def __init__(self, cost_map, weights, v_air=None, v_max=None):
        self.cost_map = cost_map
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (3,) or np.any(self.weights < 0) or not np.any(self.weights > 0):
            raise ValueError("weights must be three non-negative values with a positive sum")
        self.weights = self.weights / self.weights.sum()
        self.v_air = cost_map.v_air if v_air is None else v_air
        self.v_max = cost_map.v_max if v_max is None else v_max
        self.heuristic = compute_heuristics(cost_map, self.v_air, self.v_max)
        self.n_expanded = 0
        self.n_generated = 0
        self.max_flight_time_s = cost_map.usable_flight_time_s
        if cost_map.battery_energy_wh is None:
            self.time_to_landing = None
        elif cost_map.min_turn_radius_m is None:
            self.time_to_landing = compute_time_to_landing(cost_map, self.v_air, self.v_max)
        else:
            self.time_to_landing = compute_time_to_landing_with_turns(cost_map, self.v_air, self.v_max)

    def _state_key(self, label):
        if self.cost_map.min_turn_radius_m is None or label.parent is None:
            return label.vertex
        return (label.vertex, (label.vertex[0] - label.parent.vertex[0],
                               label.vertex[1] - label.parent.vertex[1]))

    def _successors(self, label):
        x, y = label.vertex
        height, width = self.cost_map.shape
        for dx, dy in NEIGHBOR_OFFSETS:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if label.parent is not None and not turn_is_feasible(
                    label.parent.vertex, label.vertex, nxt, self.cost_map):
                continue
            contribution = edge_objectives(label.vertex, nxt, self.cost_map,
                                           self.v_air, self.v_max)
            if contribution is not None:
                yield nxt, contribution

    def _meets_energy_constraints(self, label):
        if label.g[0] > self.max_flight_time_s + 1e-12:
            return False
        if self.time_to_landing is None:
            return True
        if isinstance(self.time_to_landing, dict):
            if self.cost_map.landing_sites[label.vertex[1], label.vertex[0]]:
                landing_time = 0.0
            elif label.parent is None:
                candidates = []
                for nxt, edge in self._successors(label):
                    direction = (nxt[0] - label.vertex[0], nxt[1] - label.vertex[1])
                    candidates.append(edge[0] + self.time_to_landing.get(
                        (nxt[0], nxt[1], *direction), np.inf))
                landing_time = min(candidates, default=np.inf)
            else:
                direction = (label.vertex[0] - label.parent.vertex[0],
                             label.vertex[1] - label.parent.vertex[1])
                landing_time = self.time_to_landing.get((*label.vertex, *direction), np.inf)
        else:
            landing_time = self.time_to_landing[label.vertex[1], label.vertex[0]]
        return bool(np.isfinite(landing_time) and
                    label.g[0] + landing_time <= self.max_flight_time_s + 1e-12)

    def solve(self):
        start, goal = tuple(self.cost_map.start), tuple(self.cost_map.goal)
        if start == goal:
            return [([start], (0.0, 0.0, 0.0))]
        if self.cost_map.is_occupied(start) or self.cost_map.is_occupied(goal):
            return []
        if not np.all(np.isfinite(self.heuristic[start[1], start[0]])):
            return []

        root = _Label(start, (0.0, 0.0, 0.0))
        if not self._meets_energy_constraints(root):
            return []
        counter = itertools.count()
        open_heap = []
        # Battery and diversion feasibility depend on elapsed flight time, not
        # solely on the scalar objective.  Keep every state label that is
        # non-dominated in (weighted cost, elapsed time), so scalarisation
        # cannot discard the only future-energy-feasible continuation.
        front = {self._state_key(root): [(0.0, 0.0)]}
        h0 = float(np.dot(self.weights, self.heuristic[start[1], start[0]]))
        heapq.heappush(open_heap, (h0, next(counter), root))
        self.n_generated = 1

        while open_heap:
            _, _, label = heapq.heappop(open_heap)
            scalar_g = float(np.dot(self.weights, label.g))
            state = self._state_key(label)
            if any(score <= scalar_g + 1e-12 and elapsed <= label.g[0] + 1e-12
                   and (score < scalar_g - 1e-12 or elapsed < label.g[0] - 1e-12)
                   for score, elapsed in front.get(state, ())):
                continue
            if not self._meets_energy_constraints(label):
                continue
            self.n_expanded += 1
            if label.vertex == goal:
                return [(self._path(label), tuple(label.g))]
            for nxt, edge in self._successors(label):
                h = self.heuristic[nxt[1], nxt[0]]
                if not np.all(np.isfinite(h)):
                    continue
                g = tuple(a + b for a, b in zip(label.g, edge))
                child = _Label(nxt, g, label)
                if not self._meets_energy_constraints(child):
                    continue
                state = self._state_key(child)
                child_scalar_g = float(np.dot(self.weights, g))
                pairs = front.get(state, [])
                if any(score <= child_scalar_g + 1e-12 and elapsed <= g[0] + 1e-12
                       for score, elapsed in pairs):
                    continue
                front[state] = [(score, elapsed) for score, elapsed in pairs
                                if not (child_scalar_g <= score + 1e-12 and
                                        g[0] <= elapsed + 1e-12)]
                front[state].append((child_scalar_g, g[0]))
                priority = child_scalar_g + float(np.dot(self.weights, h))
                heapq.heappush(open_heap, (priority, next(counter), child))
                self.n_generated += 1
        return []

    @staticmethod
    def _path(label):
        path = []
        while label is not None:
            path.append(label.vertex)
            label = label.parent
        return list(reversed(path))
