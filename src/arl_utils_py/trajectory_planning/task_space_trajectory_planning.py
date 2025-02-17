from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import Callable

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from ranges import Range, RangeDict
from scipy.spatial.transform import Rotation


def cubic_trajectory_f(t0, t1, p0, p1, v0=0.0, v1=0.0) -> Polynomial:
    """Compute cubic trajectory as a numpy Polynomial."""
    A = np.array(
        [
            [1, t0, t0**2, t0**3],
            [1, t1, t1**2, t1**3],
            [0, 1, 2 * t0, 3 * t0**2],
            [0, 1, 2 * t1, 3 * t1**2],
        ]
    )
    b = np.array([p0, p1, v0, v1])
    return Polynomial(np.linalg.solve(A, b))


def quintic_trajectory_f(t0, t1, p0, p1, v0=0.0, v1=0.0, a0=0.0, a1=0.0) -> Polynomial:
    """Compute quintic trajectory as a numpy Polynomial."""
    A = np.array(
        [
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [1, t1, t1**2, t1**3, t1**4, t1**5],
            [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
            [0, 1, 2 * t1, 3 * t1**2, 4 * t1**3, 5 * t1**4],
            [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
            [0, 0, 2, 6 * t1, 12 * t1**2, 20 * t1**3],
        ]
    )
    b = np.array([p0, p1, v0, v1, a0, a1])
    return Polynomial(np.linalg.solve(A, b))


@dataclass(frozen=True)
class TrajectoryPlannerBase:
    """TrajectoryPlannerBase

    Base class for trajectory planners. Subclasses must implement the __call__ method.

    Call to this class maps a time range to a normalized value in [0, 1] that can be used for parametric curves.
    """

    @abstractmethod
    def __call__(self, range: Range, t: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearTrajectoryPlanner(TrajectoryPlannerBase):
    def __call__(self, range: Range, t: float) -> float:
        return (t - range.start) / (range.end - range.start)


@dataclass(frozen=True)
class CubicTrajectoryPlanner(TrajectoryPlannerBase):
    def __call__(self, range: Range, t: float) -> float:
        return float(cubic_trajectory_f(range.start, range.end, 0, 1, 0, 0)(t))


@dataclass(frozen=True)
class QuinticTrajectoryPlanner(TrajectoryPlannerBase):
    def __call__(self, range: Range, t: float) -> float:
        return float(quintic_trajectory_f(range.start, range.end, 0, 1, 0, 0, 0, 0)(t))


@dataclass(frozen=True)
class TaskSpaceTrajectorySegment:
    """TaskSpaceTrajectorySegment

    t_range: Time range of this segment

    p_f_u: Parametric curve that computes position from input u from 0 to 1

    R_f_u: Parametric curve that computes orientation from input u from 0 to 1

    planner_type: Type[TrajectoryPlannerBase] how to map t_range to u
    """

    t_range: Range
    p_f_u: Callable[[float], NDArray]
    R_f_u: Callable[[float], Rotation]
    planner_type: type[TrajectoryPlannerBase]

    @property
    def p_f_t(self) -> Callable[[float], NDArray]:
        return lambda t: self.p_f_u(self.planner_type()(self.t_range, t))

    @property
    def R_f_t(self) -> Callable[[float], Rotation]:
        return lambda t: self.R_f_u(self.planner_type()(self.t_range, t))


@dataclass(frozen=True)
class TaskSpaceTrajectory:
    segments: list[TaskSpaceTrajectorySegment]

    def __post_init__(self):
        sorted_ranges = sorted(self.segments, key=lambda x: x.t_range.start)

        for segment1, segment2 in pairwise(sorted_ranges):
            # Check time continuity
            if segment1.t_range.end != segment2.t_range.start:
                if segment1.t_range.end > segment2.t_range.start:
                    raise ValueError(
                        f"Time segments overlap: {segment1.t_range} and {segment2.t_range}"
                    )
                raise ValueError(
                    f"Time segments are not continuous: {segment1.t_range} and {segment2.t_range}"
                )

            # Check position continuity
            p_end = segment1.p_f_t(segment1.t_range.end)
            p_start = segment2.p_f_t(segment2.t_range.start)
            if not np.allclose(p_end, p_start):
                raise ValueError(
                    f"Position discontinuity at t={segment1.t_range.end}: {p_end} != {p_start}"
                )

            # Check orientation continuity
            Q_end = segment1.R_f_t(segment1.t_range.end)
            Q_start = segment2.R_f_t(segment2.t_range.start)
            if not np.allclose(
                Q_end.as_quat(), Q_start.as_quat()
            ):  # Compare quaternions
                raise ValueError(
                    f"Orientation discontinuity at t={segment1.t_range.end}: {Q_end.as_quat()} != {Q_start.as_quat()}"
                )

        # Ensure the last range is closed `[start, end]`
        last_range = sorted_ranges[-1].t_range
        if last_range.end not in last_range:
            raise ValueError(f"Last time segment must be closed: {last_range}")

    @cached_property
    def __f_dict(self) -> RangeDict:
        return RangeDict(
            {
                segment.t_range: (segment.p_f_t, segment.R_f_t)
                for segment in self.segments
            }
        )

    def __call__(self, t: float) -> tuple[NDArray, Rotation]:
        p_func, R_func = self.__f_dict[t]
        return p_func(t), R_func(t)
