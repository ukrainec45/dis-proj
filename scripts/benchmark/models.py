"""Common result objects shared by benchmark methods and reporters."""

from dataclasses import dataclass, field


@dataclass
class PlannerResult:
    """Output of one planner on one immutable :class:`CostMap`."""

    method: str
    solutions: list[tuple[list[tuple[int, int]], tuple[float, float, float]]]
    runtime_ms: float
    n_expanded: int | None = None
    n_generated: int | None = None
    details: dict = field(default_factory=dict)

    @property
    def feasible(self):
        return bool(self.solutions)
