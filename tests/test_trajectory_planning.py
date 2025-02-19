from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ranges import Range
from scipy.spatial.transform import Rotation

from arl_utils_py.trajectory_planning.task_space_trajectory_planning import (
    CubicTrajectoryPlanner,
    LinearTrajectoryPlanner,
    QuinticTrajectoryPlanner,
    TaskSpaceTrajectory,
    TaskSpaceTrajectorySegment,
    TrapezoidalTrajectoryPlanner,
)


def test_trajectory_planning_all(tmp_path: Path):
    T0 = 0
    T1 = 2
    T2 = 4

    def seg1_p_func(u):
        return np.array([1, 1, 1])

    def seg1_R_func(u):
        return Rotation.from_euler("zyz", [0, 0, 0])

    def seg2_p_func(u):
        return np.array([1, 1, 1]) + u * np.array([1, 2, 3])

    def seg2_R_func(u):
        return Rotation.from_euler("zyz", [u * np.pi / 2, 0, 0])

    segment_params = [
        (
            Range(T0, T1),
            seg1_p_func,
            seg1_R_func,
        ),
        (
            Range(T1, T2, include_end=True),
            seg2_p_func,
            seg2_R_func,
        ),
    ]

    linear_trajectory_func = TaskSpaceTrajectory(
        [
            TaskSpaceTrajectorySegment(*param, LinearTrajectoryPlanner())
            for param in segment_params
        ]
    )
    cubic_trajectory_func = TaskSpaceTrajectory(
        [
            TaskSpaceTrajectorySegment(*param, CubicTrajectoryPlanner())
            for param in segment_params
        ]
    )
    quintic_trajectory_func = TaskSpaceTrajectory(
        [
            TaskSpaceTrajectorySegment(*param, QuinticTrajectoryPlanner())
            for param in segment_params
        ]
    )
    trapezoidal_trajectory_func = TaskSpaceTrajectory(
        [
            TaskSpaceTrajectorySegment(*param, planner)
            for param, planner in zip(
                segment_params,
                [
                    TrapezoidalTrajectoryPlanner(),
                    TrapezoidalTrajectoryPlanner(v_max_unsigned=1 / (T2 - T1) * 1.2),
                ],
            )
        ]
    )

    t_array = np.linspace(T0, T2, T_NUM := 200)
    p_list_linear, R_list_linear = zip(*[linear_trajectory_func(t) for t in t_array])
    plt.plot(t_array, p_list_linear, label="linear")
    p_list_cubic, R_list_cubic = zip(*[cubic_trajectory_func(t) for t in t_array])
    plt.plot(t_array, p_list_cubic, label="cubic")
    p_list_quintic, R_list_quintic = zip(*[quintic_trajectory_func(t) for t in t_array])
    plt.plot(t_array, p_list_quintic, label="quintic")
    p_list_trapezoidal, R_list_trapezoidal = zip(
        *[trapezoidal_trajectory_func(t) for t in t_array]
    )
    plt.plot(t_array, p_list_trapezoidal, label="trapezoidal")

    plt.legend()
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("position")
    plt.title("Linear/Cubic/Quintic/Trapezoidal Trajectory Planning Results")
    plt.savefig(tmp_path / "test_trajectory_planning_all.png")
    plt.close()

    for (p_list_1, R_list_1), (p_list_2, R_list_2) in combinations(
        [
            (p_list_linear, R_list_linear),
            (p_list_cubic, R_list_cubic),
            (p_list_quintic, R_list_quintic),
            (p_list_trapezoidal, R_list_trapezoidal),
        ],
        2,
    ):
        # Start Point
        assert np.allclose(p_list_1[0], p_list_2[0])
        assert np.allclose(R_list_1[0].as_matrix(), R_list_2[0].as_matrix())
        # End Point
        assert np.allclose(p_list_1[-1], p_list_2[-1])
        assert np.allclose(R_list_1[-1].as_matrix(), R_list_2[-1].as_matrix())
        # T1 point
        index = np.argwhere(
            np.isclose(t_array, T1, rtol=(T2 - T0) / T_NUM / 2)
        ).flatten()[0]
        assert np.allclose(p_list_1[index], p_list_2[index])
        assert np.allclose(R_list_1[index].as_matrix(), R_list_2[index].as_matrix())
