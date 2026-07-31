"""EMOA*-Late-BS: exact multi-objective A* with late dominance checking and
binary-search fronts.

Implements the EMOA* framework from Ren et al., "EMOA*: A Framework for
Search-Based Multi-Objective Path Planning", Artificial Intelligence (2024):
    * Algorithm 3  - EMOA*-Late (lazy dominance checks)
    * Algorithm 6  - BS-Check  (dominance check on a lex-sorted list)
    * Algorithm 7  - BS-Filter (filter + insert on a lex-sorted list)

Structure
---------
Per-vertex fronts ``F(v)`` (v != goal) store lex-sorted, cost-unique
non-dominated *truncated* cost vectors ``Trunc(g) = (g_2, ..., g_M)`` and are
used only for dominance pruning. Truncation is valid because all objectives use
consistent heuristics and labels are expanded in lexicographic order of
``f = g + h``, so labels expanded at the same vertex have non-decreasing
``g_1``; hence a closed label whose truncated vector weakly dominates a checked
label's truncated vector also dominates it in the full vector.

The Pareto-optimal solutions are kept in a separate list ``sols`` with
**full-vector** dominance (cost-unique). This is essential: a later-extracted
goal label can truncated-dominate an already-closed solution with a *smaller*
``g_1``, and removing that solution (as a truncated front filter would) would
drop a genuine Pareto-optimal path. Front filters are therefore never applied
to the solution set; instead ``sols`` is maintained as a non-dominated set over
full cost vectors, so a new goal label also removes any stored solution it
dominates.
"""

import heapq
import itertools
from bisect import bisect_right

import numpy as np

from .edge_costs import NEIGHBOR_OFFSETS, edge_objectives, turn_is_feasible
from .heuristics import (compute_heuristics, compute_time_to_landing,
                         compute_time_to_landing_with_turns)


class Label:
    __slots__ = ("vertex", "g", "f", "parent")

    def __init__(self, vertex, g, f, parent):
        self.vertex = vertex
        self.g = g  # full cost vector from start (M components)
        self.f = f  # g + h
        self.parent = parent

    def truncated_g(self):
        return self.g[1:]


def bs_check(B, b):
    """Algorithm 6: True iff some vector in the lex-sorted list B weakly dominates b."""
    i = bisect_right(B, b)
    for bp in B[:i]:
        if all(bpk <= bk for bpk, bk in zip(bp, b)):
            return True
    return False


def bs_filter(B, b):
    """Algorithm 7: remove vectors weakly dominated by b, insert b, keep lex-sorted."""
    i = bisect_right(B, b)
    start = i if i > 0 else 0
    kept = [bp for bp in B[start:] if not all(bk <= bpk for bk, bpk in zip(b, bp))]
    return B[:i] + [b] + kept


class EmoaStarLateBS:
    """Exact Pareto-front solver for the 2.5D grid planning problem.

    Parameters
    ----------
    cost_map : CostMap
        Grid layers, start and goal cells, ``v_air``/``v_max`` defaults.
    v_air, v_max : float, optional
        Override the UAV speed limits from ``cost_map``.
    heuristic : np.ndarray or callable, optional
        [H, W, M] lower-bound heuristic (default: backward Dijkstra).
    """

    def __init__(self, cost_map, v_air=None, v_max=None, heuristic=None, record=False):
        self.cost_map = cost_map
        self.v_air = v_air if v_air is not None else cost_map.v_air
        self.v_max = v_max if v_max is not None else cost_map.v_max
        if self.v_air is None or self.v_max is None:
            raise ValueError("v_air and v_max are required")
        if self.v_air < cost_map.v_air_min_mps:
            raise ValueError("v_air must be at least v_air_min_mps")
        if (cost_map.v_air_max_mps is not None and
                self.v_air > cost_map.v_air_max_mps):
            raise ValueError("v_air must not exceed v_air_max_mps")
        if self.v_max <= 0:
            raise ValueError("v_max must be positive")
        if heuristic is None:
            heuristic = compute_heuristics(cost_map, self.v_air, self.v_max)
        self.heuristic = heuristic
        self.start = tuple(cost_map.start)
        self.goal = tuple(cost_map.goal)
        self.M = 3
        self.max_flight_time_s = cost_map.usable_flight_time_s
        if cost_map.battery_energy_wh is None:
            self.time_to_landing = None
        elif cost_map.min_turn_radius_m is None:
            self.time_to_landing = compute_time_to_landing(cost_map, self.v_air, self.v_max)
        else:
            self.time_to_landing = compute_time_to_landing_with_turns(cost_map, self.v_air, self.v_max)

        # With a turn-radius constraint, an incoming heading is part of the
        # state.  Dominance across headings would be unsound because it could
        # remove the only label able to make the next feasible turn.
        self.front = {}    # search state -> lex-sorted truncated cost vectors
        self.sols = []     # goal Labels (full costs), cost-unique, never filtered
        self.n_expanded = 0
        self.n_generated = 0

        # Optional trace of the search for visualisation (see scripts/visualize.py).
        # Each event: {"kind", "vertex", "f", "g", "n_open", "front_size",
        #              "n_sols", "front_vertex", "prev_sols", "reason"}.
        self.record = bool(record)
        self.events = []

    def _emit(self, **kw):
        if self.record:
            self.events.append(kw)

    # -- heuristic lookup ----------------------------------------------------
    def _h_of(self, v):
        h = self.heuristic
        if callable(h):
            return tuple(float(x) for x in h(v))
        return tuple(float(x) for x in h[v[1], v[0]])

    # -- successors ----------------------------------------------------------
    def _successors(self, l):
        x, y = l.vertex
        H, W = self.cost_map.dem.shape
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if l.parent is not None and not turn_is_feasible(
                        l.parent.vertex, l.vertex, (nx, ny), self.cost_map):
                    continue
                yield (nx, ny), edge_objectives(
                    l.vertex, (nx, ny), self.cost_map, self.v_air, self.v_max
                )

    def _state_key(self, l):
        if self.cost_map.min_turn_radius_m is None or l.parent is None:
            return l.vertex
        return (l.vertex, (l.vertex[0] - l.parent.vertex[0],
                           l.vertex[1] - l.parent.vertex[1]))

    # -- dominance checks -----------------------------------------------------
    def _is_dom_by_front(self, l):
        B = self.front.get(self._state_key(l))
        return B is not None and bs_check(B, l.truncated_g())

    def _is_dom_by_sol(self, l):
        return any(all(s_comp <= f_comp for s_comp, f_comp in zip(s.g, l.f))
                   for s in self.sols)

    def _pruned(self, l):
        return self._is_dom_by_front(l) or self._is_dom_by_sol(l)

    def _meets_energy_constraints(self, l):
        """Check route budget and optional emergency-landing reserve.

        The constant-power energy model makes energy proportional to ``g[0]``;
        therefore this hard constraint is compatible with the existing exact
        three-objective dominance rules.
        """
        if l.g[0] > self.max_flight_time_s + 1e-12:
            return False
        if self.time_to_landing is None:
            return True
        if isinstance(self.time_to_landing, dict):
            if self.cost_map.landing_sites[l.vertex[1], l.vertex[0]]:
                t_land = 0.0
            elif l.parent is None:
                candidates = []
                for u, c in self._successors(l):
                    if c is not None:
                        direction = (u[0] - l.vertex[0], u[1] - l.vertex[1])
                        candidates.append(c[0] + self.time_to_landing.get(
                            (u[0], u[1], *direction), np.inf
                        ))
                t_land = min(candidates, default=np.inf)
            else:
                direction = (l.vertex[0] - l.parent.vertex[0],
                             l.vertex[1] - l.parent.vertex[1])
                t_land = self.time_to_landing.get((*l.vertex, *direction), np.inf)
        else:
            t_land = self.time_to_landing[l.vertex[1], l.vertex[0]]
        return bool(np.isfinite(t_land) and l.g[0] + t_land <= self.max_flight_time_s + 1e-12)

    def _prune_reason(self, l):
        """Return ("pruned", "front"|"sol") or (False, None)."""
        if self._is_dom_by_front(l):
            return True, "front"
        if self._is_dom_by_sol(l):
            return True, "sol"
        return False, None

    def _add_sol(self, l):
        """Insert a goal label into the non-dominated solution set (full vectors).

        The label already passed ``_is_dom_by_sol`` (not dominated by any stored
        solution). Stored solutions dominated by the new label are removed, so
        ``sols`` always stays a mutually non-dominated, cost-unique set.
        """
        removed = [s.g for s in self.sols
                   if all(l.g[j] <= s.g[j] for j in range(self.M))]
        keep = [s for s in self.sols
                if not all(l.g[j] <= s.g[j] for j in range(self.M))]
        self.sols = keep + [l]
        if self.record:
            self.events.append({
                "kind": "add_sol", "vertex": l.vertex, "f": l.f, "g": l.g,
                "n_open": 0, "front_size": 0, "front_vertex": l.vertex,
                "n_sols": len(self.sols),
                "evicted": [tuple(g) for g in removed],
                "prev_sols": [tuple(s.g) for s in keep],
            })

    # -- search ----------------------------------------------------------------
    def solve(self):
        """Return the exact Pareto front as a list of ``(path, cost_vector)``.

        ``path`` is the ordered waypoint sequence (including start and goal);
        ``cost_vector`` is the non-dominated ``(f1, f2, f3)`` of that path.
        Returns an empty list when no feasible path exists.
        """
        start, goal = self.start, self.goal
        M = self.M
        cm = self.cost_map

        if start == goal:
            return [([start], (0.0,) * M)]
        if cm.is_occupied(start) or cm.is_occupied(goal):
            return []
        if not all(np.isfinite(self._h_of(start))):
            return []

        g0 = (0.0,) * M
        f0 = tuple(a + b for a, b in zip(g0, self._h_of(start)))
        root = Label(start, g0, f0, None)
        if not self._meets_energy_constraints(root):
            return []
        open_heap = []
        counter = itertools.count()
        heapq.heappush(open_heap, (f0, next(counter), root))
        self.n_generated += 1
        self._emit(kind="root", vertex=start, f=f0, g=g0,
                   n_open=len(open_heap), front_size=0, n_sols=0)

        while open_heap:
            _, _, l = heapq.heappop(open_heap)
            self.n_expanded += 1
            pruned, reason = self._prune_reason(l)
            if not pruned and not self._meets_energy_constraints(l):
                pruned, reason = True, "energy"
            self._emit(kind="pop", vertex=l.vertex, f=l.f, g=l.g,
                       pruned=pruned, reason=reason,
                       n_open=len(open_heap),
            front_size=len(self.front.get(self._state_key(l), [])),
                       front_vertex=l.vertex,
                       n_sols=len(self.sols),
                       prev_sols=[tuple(s.g) for s in self.sols])
            if pruned:  # late dominance check
                continue

            if l.vertex == goal:
                self._add_sol(l)
                continue

            state = self._state_key(l)
            B = self.front.get(state, [])
            self.front[state] = bs_filter(B, l.truncated_g())
            self._emit(kind="expand", vertex=l.vertex, f=l.f, g=l.g,
                       n_open=len(open_heap),
                       front_size=len(self.front[state]),
                       front_vertex=l.vertex,
                       n_sols=len(self.sols))

            for u, c in self._successors(l):
                if c is None:
                    continue
                h_u = self._h_of(u)
                if not all(np.isfinite(h_u)):
                    continue  # u cannot reach the goal
                g2 = tuple(a + b for a, b in zip(l.g, c))
                f2 = tuple(a + b for a, b in zip(g2, h_u))
                child = Label(u, g2, f2, l)
                self.n_generated += 1
                cpruned, creason = self._prune_reason(child)
                if not cpruned and not self._meets_energy_constraints(child):
                    cpruned, creason = True, "energy"
                if cpruned:
                    self._emit(kind="gen_pruned", vertex=u, f=f2, g=g2,
                               reason=creason, parent=l.vertex,
                               n_open=len(open_heap),
                               front_size=len(self.front.get(self._state_key(child), [])),
                               front_vertex=u,
                               n_sols=len(self.sols))
                    continue
                heapq.heappush(open_heap, (f2, next(counter), child))
                self._emit(kind="gen", vertex=u, f=f2, g=g2, parent=l.vertex,
                           n_open=len(open_heap),
                           front_size=len(self.front.get(self._state_key(child), [])),
                           front_vertex=u,
                           n_sols=len(self.sols))

        return self._extract_solutions()

    def _extract_solutions(self):
        sols = []
        for l in self.sols:
            path = []
            node = l
            while node is not None:
                path.append(node.vertex)
                node = node.parent
            path.reverse()
            sols.append((path, tuple(l.g)))
        sols.sort(key=lambda item: item[1])
        return sols
