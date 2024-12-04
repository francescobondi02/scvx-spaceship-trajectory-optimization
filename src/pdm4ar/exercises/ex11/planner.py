# PUSH 2
from dataclasses import dataclass, field
from typing import Union
import numpy as np

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *

# from pdm4ar.exercises_def.ex07.structures import Constraints
from pdm4ar.exercises_def.ex11.goal import DockingTarget, SpaceshipTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import matplotlib.pyplot as plt
import os
from dg_commons.sim.models.obstacles import StaticObstacle


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-3  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    X_linear: NDArray
    U_int: NDArray
    p_int: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        box: StaticObstacle,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.n_planets = len(self.planets)
        self.satellites = satellites
        self.sg = sg
        self.sp = sp
        self.r_buffer_stat_obs = self.sg.l_c + self.sg.l_f + 0.1  # Buffer for static obstacles
        self.epsilon = 1e-4  # tolerance for final state
        self.box = box
        self.docking = False

        # Solver Parameters
        self.params = SolverParameters()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

    def compute_trajectory(
        self, init_state: SpaceshipState, goal: SpaceshipTarget | DockingTarget
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        goal_state = goal.target
        if isinstance(goal, DockingTarget):
            # Compute points for docking correctly (using constraints later)
            self.docking = True
            A, B, C, A1, A2, half_ang_aper = goal.get_landing_constraint_points()
            center_x = (A1[0] + A2[0]) / 2
            center_y = (A1[1] + A2[1]) / 2

            direction_angle = np.arctan2(A2[1] - A1[1], A2[0] - A1[0])

            x_goal = center_x + np.cos(direction_angle + np.pi / 2) * (self.sg.l_r)  # is A[0]
            y_goal = center_y + np.sin(direction_angle + np.pi / 2) * (self.sg.l_r)  # is A[1]

            goal_state.x = x_goal
            goal_state.y = y_goal
            goal_state.x = (A[0] * 2 + x_goal) / 3
            goal_state.y = (A[1] * 2 + y_goal) / 3
            goal_state.psi = np.arctan2(A2[1] - A1[1], A2[0] - A1[0]) + np.pi / 2

            len_A2B = np.sqrt((A2[0] - B[0]) ** 2 + (A2[1] - B[1]) ** 2)
            D = np.zeros(2)
            E = np.zeros(2)
            D[0] = center_x + 3 / 3 * len_A2B * np.cos(direction_angle + np.pi / 2)
            D[1] = center_y + 3 / 3 * len_A2B * np.sin(direction_angle + np.pi / 2)

            E[0] = center_x + 4 / 3 * len_A2B * np.cos(direction_angle + np.pi / 2)
            E[1] = center_y + 4 / 3 * len_A2B * np.sin(direction_angle + np.pi / 2)

            print(f"Goal state dock: {goal_state}")

            # Visualize the docking points
            # self.docking_points_visualizer(A, B, C, A1, A2, center_x, center_y, x_goal, y_goal, D, E)

        # Initialize the problem parameters
        if self.docking:
            self.problem_parameters["goal_minus_1"].value = np.array((D[0], D[1], goal_state.psi))
            self.problem_parameters["goal_minus_2"].value = np.array((E[0], E[1], goal_state.psi))

        self.init_state = init_state
        self.goal_state = goal_state
        self.problem_parameters["tr_radius"].value = self.params.tr_radius
        self.problem_parameters["init_state"].value = [
            self.init_state.x,
            self.init_state.y,
            self.init_state.psi,
            self.init_state.vx,
            self.init_state.vy,
            self.init_state.dpsi,
            self.init_state.delta,
            self.init_state.m,
        ]
        self.problem_parameters["goal_state"].value = [
            self.goal_state.x,
            self.goal_state.y,
            self.goal_state.psi,
            self.goal_state.vx,
            self.goal_state.vy,
            self.goal_state.dpsi,
        ]

        (
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        ) = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

        for i in range(self.params.max_iterations):
            print(f"SCvx Iteration {i + 1}")
            self._convexification()
            try:
                self.error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            if self._check_convergence():
                mycmds, mystates = self._extract_seq_from_array()
                break

            X_bar, U_bar, p_bar, tr_radius = self._update_trust_region()

            self.problem_parameters["X_bar"].value = X_bar.copy()
            self.problem_parameters["U_bar"].value = U_bar.copy()
            self.problem_parameters["p_bar"].value = p_bar.copy()
            self.problem_parameters["tr_radius"].value = tr_radius

        if i == self.params.max_iterations - 1:
            mycmds, mystates = self._extract_seq_from_array()

        print(f"Final time: {self.variables['p'].value}")

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        # Number of steps in the trajectory
        K = self.params.K

        # Initialize state trajectory (X_bar)
        X = np.zeros((self.spaceship.n_x, K))  # State trajectory

        # Interpolate position and orientation
        X[0, :] = np.linspace(self.init_state.x, self.goal_state.x, K)  # x position
        X[1, :] = np.linspace(self.init_state.y, self.goal_state.y, K)  # y position

        # Handle angles correctly (wrap around) correct difference between angles start and finish
        delta_psi = (self.goal_state.psi - self.init_state.psi + np.pi) % (2 * np.pi) - np.pi
        X[2, :] = self.init_state.psi + np.linspace(0, delta_psi, K)  # orientation (psi)

        # Set velocities (vx, vy) and angular velocity (dpsi) to 0 initially
        X[3, :] = 0  # vx
        X[4, :] = 0  # vy
        X[5, :] = 0  # dpsi

        # Set thrust direction angle (delta) to 0 initially
        X[6, :] = 0  # delta

        # Interpolate mass, decrement linearly
        initial_mass = self.init_state.m
        final_mass = self.sp.m_v
        X[7, :] = np.linspace(initial_mass, final_mass, K)

        # Initialize control inputs (U_bar)
        U = np.zeros((self.spaceship.n_u, K))  # Control inputs
        # Initialize control inputs (U_bar) to zero
        U[0, :] = 0  # Thrust is initially zero
        U[1, :] = 0  # No change in thrust angle (ddelta)

        # Initialize parameters (p_bar)
        p = np.zeros(self.spaceship.n_p)  # Parameters
        p = np.array([30])  # Time estimate based on average speed

        return X, U, p

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        if self.box is not None:
            box_bounds = self.box.shape.bounds
            min_x, min_y, max_x, max_y = box_bounds

        # Basic constraints
        constraints = [
            # Constraints on time
            self.variables["p"] >= 0,
            # Boundary constraint
            self.variables["X"][0, :] >= min_x + self.sg.l_f + self.sg.l_c,
            self.variables["X"][0, :] <= max_x - self.sg.l_f - self.sg.l_c,
            self.variables["X"][1, :] >= min_y + self.sg.l_f + self.sg.l_c,
            self.variables["X"][1, :] <= max_y - self.sg.l_f - self.sg.l_c,
            # Constraints on admissible states and inputs #55c
            self.variables["X"][6, :] >= self.spaceship.sp.delta_limits[0],
            self.variables["X"][6, :] <= self.spaceship.sp.delta_limits[1],
            self.variables["U"][0, :] >= self.spaceship.sp.thrust_limits[0],
            self.variables["U"][0, :] <= self.spaceship.sp.thrust_limits[1],
            self.variables["U"][1, :] >= self.spaceship.sp.ddelta_limits[0],
            self.variables["U"][1, :] <= self.spaceship.sp.ddelta_limits[1],
            # Constraints su condizione iniziale e finale
            self.variables["X"][:, 0] - self.problem_parameters["init_state"] == 0,  # 55e
            self.variables["X"][0:6, -1] - self.problem_parameters["goal_state"] + self.variables["nu_tc"] == 0,  # 55f
            self.variables["U"][:, 0] == 0,
            self.variables["U"][:, -1] == 0,
        ]

        # Constraints on dynamics #55b
        constraints += self._get_dynamics_constraints()

        # Constraints su obstacle avoidance #55d
        # Static
        constraints += self._get_static_constraints()

        # Constraints on trust region #55g
        constraints += self._get_thrust_region_constraints()

        # Satellites constraints (dynamic)
        constraints += self._get_dynamic_constraints()

        return constraints

    def _get_dynamics_constraints(self) -> list[cvx.Constraint]:
        """
        Dynamics constraints.
        """
        constraints = []
        for i in range(self.params.K - 1):
            E = np.eye(self.spaceship.n_x)

            constraints.append(
                self.variables["X"][:, i + 1]
                == self.problem_parameters["A_bar"][i] @ self.variables["X"][:, i]
                + self.problem_parameters["B_minus_bar"][i] @ self.variables["U"][:, i]
                + self.problem_parameters["B_plus_bar"][i] @ self.variables["U"][:, i + 1]
                + self.problem_parameters["F_bar"][i] @ self.variables["p"]
                + self.problem_parameters["r_bar"][i]
                + E @ self.variables["nu"][:, i]
            )
        return constraints

    def _get_thrust_region_constraints(self) -> list[cvx.Constraint]:
        """
        Thrust region constraints.
        """
        constraints = []
        constraints.append(
            cvx.norm(self.variables["X"] - self.problem_parameters["X_bar"], 1)
            + cvx.norm(self.variables["U"] - self.problem_parameters["U_bar"], 1)
            + cvx.norm(self.variables["p"] - self.problem_parameters["p_bar"], 1)
            - self.problem_parameters["tr_radius"]
            <= 0
        )
        return constraints

    def _get_static_constraints(self) -> list[cvx.Constraint]:
        """
        Constraints for static obstacles.
        """
        constraints = []
        if self.planets is not None:
            for player_name in self.planets:
                xp, yp = self.planets[player_name].center
                r = self.planets[player_name].radius + self.r_buffer_stat_obs

                for k in range(self.params.K):
                    xk_bar = self.problem_parameters["X_bar"][0, k]  # Nominal x-coordinate at step k
                    yk_bar = self.problem_parameters["X_bar"][1, k]  # Nominal y-coordinate at step k

                    xk = self.variables["X"][0, k]  # Actual x-coordinate at step k
                    yk = self.variables["X"][1, k]  # Actual y-coordinate at step k

                    # Use the nominal values to compute the constants for the linear constraint
                    minus_2_x_diff = -2 * (xk_bar - xp)
                    minus_2_y_diff = -2 * (yk_bar - yp)

                    # Compute rk_prime for the linearized constraint
                    rk_prime = (
                        -((xk_bar - xp) ** 2)
                        - ((yk_bar - yp) ** 2)
                        + r**2
                        + 2 * (xk_bar - xp) * xk_bar
                        + 2 * (yk_bar - yp) * yk_bar
                    )

                    # Append the linearized constraint
                    constraints.append(
                        (minus_2_x_diff * xk + minus_2_y_diff * yk + rk_prime) <= self.variables["nu_s"][k]
                    )
        return constraints

    def _get_dynamic_constraints(self) -> list[cvx.Constraint]:
        constraints = []
        if (len(self.satellites)) > 0:
            index_satellite = 0
            for planet_name, satellites_params in self.satellites.items():
                for k in range(self.params.K):
                    # Get the linearized parameters for this satellite at this timestep
                    G_sat = self.problem_parameters["G_sat"][index_satellite, k]
                    C_sat_x = self.problem_parameters["C_sat"][index_satellite * 2, k]
                    C_sat_y = self.problem_parameters["C_sat"][index_satellite * 2 + 1, k]
                    r_prime_sat = self.problem_parameters["r_prime_sat"][index_satellite, k]
                    constraints.append(
                        C_sat_x * self.variables["X"][0, k]
                        + C_sat_y * self.variables["X"][1, k]
                        + G_sat * self.variables["p"] * k / (self.params.K - 1)
                        + r_prime_sat
                        <= self.variables["nu_s_sat"][k]
                    )
                index_satellite += 1
        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective cost for SCvx.
        """
        weight_dfg = 1
        weight_ctrl = 100
        weight_xfg = 1
        terminal_cost = (
            weight_dfg * cvx.norm(self.variables["X"][0:6, -1] - self.problem_parameters["goal_state"], 2)
            + self.params.weight_p @ self.variables["p"]
            + self.params.lambda_nu * cvx.norm(self.variables["nu_tc"], 1)
            #  + self.params.lambda_nu * cvx.norm(self.variables["nu_ic"], 1)
            + self.params.lambda_nu * cvx.norm(self.variables["nu"], 1)
            + self.params.lambda_nu * cvx.norm(self.variables["nu_s"], 1)
            + self.params.lambda_nu * cvx.norm(self.variables["nu_s_sat"], 1)
        )

        stage_cost = 0
        for i in range(self.params.K - 1):
            stage_cost += weight_ctrl * cvx.norm(self.variables["U"][:, i + 1] - self.variables["U"][:, i], 1)
            stage_cost += weight_xfg * cvx.norm(self.problem_parameters["goal_state"] - self.variables["X"][0:6, i], 1)

        objective = terminal_cost + stage_cost

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        X_bar = self.problem_parameters["X_bar"].value
        U_bar = self.problem_parameters["U_bar"].value
        p_bar = self.problem_parameters["p_bar"].value

        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(X_bar, U_bar, p_bar)

        self.problem_parameters["init_state"].value = X_bar[:, 0]

        for k in range(self.params.K - 1):
            self.problem_parameters["A_bar"][k].value = np.reshape(
                A_bar[:, k], (self.spaceship.n_x, self.spaceship.n_x), order="F"
            ).copy()
            self.problem_parameters["B_plus_bar"][k].value = np.reshape(
                B_plus_bar[:, k], (self.spaceship.n_x, self.spaceship.n_u), order="F"
            ).copy()
            self.problem_parameters["B_minus_bar"][k].value = np.reshape(
                B_minus_bar[:, k], (self.spaceship.n_x, self.spaceship.n_u), order="F"
            ).copy()
            self.problem_parameters["F_bar"][k].value = np.reshape(
                F_bar[:, k], (self.spaceship.n_x, self.spaceship.n_p), order="F"
            ).copy()
            self.problem_parameters["r_bar"][k].value = r_bar[:, k].copy()

        # Update satellite constraints
        if (len(self.satellites)) > 0:
            G_sat = np.empty([1 * len(self.satellites), self.params.K])
            C_sat = np.empty([2 * len(self.satellites), self.params.K])
            r_prime_sat = np.empty([1 * len(self.satellites), self.params.K])
            index_planet = 0
            for planet_name, satellite_params in self.satellites.items():
                tau = satellite_params.tau
                orbit_r = satellite_params.orbit_r
                omega = satellite_params.omega
                radius = satellite_params.radius
                planet_center = self.planets[planet_name.split("/")[0]].center
                planet_center = np.array(planet_center)
                for k in range(self.params.K):
                    # Compute problem parameters
                    t_k = self.problem_parameters["p_bar"].value[0] * k / (self.params.K - 1)
                    C_sat_x = -2 * (
                        self.problem_parameters["X_bar"].value[0, k]
                        - planet_center[0]
                        - orbit_r * np.cos(omega * t_k + tau)
                    )
                    C_sat_y = -2 * (
                        self.problem_parameters["X_bar"].value[1, k]
                        - planet_center[1]
                        - orbit_r * np.sin(omega * t_k + tau)
                    )
                    G_sat_term_1 = (
                        -2
                        * (
                            self.problem_parameters["X_bar"].value[0, k]
                            - planet_center[0]
                            - orbit_r * np.cos(omega * t_k + tau)
                        )
                        * (-orbit_r * omega * np.sin(omega * t_k + tau))
                    )
                    G_sat_term_2 = (
                        -2
                        * (
                            self.problem_parameters["X_bar"].value[1, k]
                            - planet_center[1]
                            - orbit_r * np.sin(omega * t_k + tau)
                        )
                        * (orbit_r * omega * np.cos(omega * t_k + tau))
                    )
                    G_sat_c = G_sat_term_1 + G_sat_term_2
                    # R_prime conservative factor
                    r = radius + self.r_buffer_stat_obs
                    r_prime_sat_c = (
                        -(
                            (
                                self.problem_parameters["X_bar"].value[0, k]
                                - planet_center[0]
                                - orbit_r * np.cos(omega * t_k + tau)
                            )
                            ** 2
                        )
                        - (
                            (
                                self.problem_parameters["X_bar"].value[1, k]
                                - planet_center[1]
                                - orbit_r * np.sin(omega * t_k + tau)
                            )
                            ** 2
                        )
                        + r**2
                        - C_sat_x * self.problem_parameters["X_bar"].value[0, k]
                        - C_sat_y * self.problem_parameters["X_bar"].value[1, k]
                        - G_sat_c * t_k
                    )

                    # Update values
                    C_sat[index_planet * 2, k] = C_sat_x
                    C_sat[index_planet * 2 + 1, k] = C_sat_y
                    G_sat[index_planet, k] = G_sat_c
                    r_prime_sat[index_planet, k] = r_prime_sat_c
                index_planet += 1

            # Update parameters
            self.problem_parameters["C_sat"].value = C_sat
            self.problem_parameters["G_sat"].value = G_sat
            self.problem_parameters["r_prime_sat"].value = r_prime_sat

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        # Termination condition 2 from paper

        param_diff_norm = np.linalg.norm(self.variables["p"].value - self.problem_parameters["p_bar"].value, 2)

        state_diff_norms = np.vstack(
            [
                np.linalg.norm(self.variables["X"].value[:, k] - self.problem_parameters["X_bar"].value[:, k], 2)
                for k in range(self.params.K)
            ]
        )
        max_state_diff_norm = np.max(state_diff_norms)



        if param_diff_norm + max_state_diff_norm <= self.params.stop_crit:
            return True

        return False

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        J_lambda_bar = self._compute_mathcal_J(
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        )
        J_lambda_star = self._compute_mathcal_J(
            self.variables["X"].value, self.variables["U"].value, self.variables["p"].value
        )
        
        L_star = self.error
        actual_change = J_lambda_bar - J_lambda_star
        predicted_change = J_lambda_bar - L_star

        rho = actual_change / predicted_change

       
        tr_radius = self.problem_parameters["tr_radius"].value  
        if rho < self.params.rho_0:
            tr_radius = max(tr_radius / self.params.alpha, self.params.min_tr_radius)
            X_bar = self.problem_parameters["X_bar"].value.copy()
            U_bar = self.problem_parameters["U_bar"].value.copy()
            p_bar = self.problem_parameters["p_bar"].value.copy()
        elif (rho < self.params.rho_1) and (rho >= self.params.rho_0):
            tr_radius = max(tr_radius / self.params.alpha, self.params.min_tr_radius)
            X_bar = self.variables["X"].value.copy()
            U_bar = self.variables["U"].value.copy()
            p_bar = self.variables["p"].value.copy()
        elif (rho < self.params.rho_2) and (rho >= self.params.rho_1):
            X_bar = self.variables["X"].value.copy()
            U_bar = self.variables["U"].value.copy()
            p_bar = self.variables["p"].value.copy()
        else:
            tr_radius = min(tr_radius * self.params.beta, self.params.max_tr_radius)
            X_bar = self.variables["X"].value.copy()
            U_bar = self.variables["U"].value.copy()
            p_bar = self.variables["p"].value.copy()

        print(f"Updated trust region radius: {tr_radius}")

        return X_bar, U_bar, p_bar, tr_radius

    def _compute_mathcal_J(self, x, u, p) -> float:
        weight_dfg = 1
        weight_ctrl = 100
        weight_xfg = 1

        X_nl = self.integrator.integrate_nonlinear_piecewise(x, u, p)
        delta_k = x - X_nl
        terminal_cost = (
            weight_dfg * np.linalg.norm(x[0:6, -1] - self.problem_parameters["goal_state"].value, 2)
            + self.params.weight_p * p[0]
            # + self.params.lambda_nu * np.linalg.norm(x[:, 0] - self.problem_parameters["init_state"].value, 1)
            + self.params.lambda_nu * np.linalg.norm(x[0:6, -1] - self.problem_parameters["goal_state"].value, 1)
            + self.params.lambda_nu * np.linalg.norm(delta_k, 1)
            + self.params.lambda_nu * np.linalg.norm(self._compute_s(x, p), 1)
            + self.params.lambda_nu * np.linalg.norm(self._compute_s_sat(x, p), 1)
        )

        stage_cost = 0
        for i in range(self.params.K - 1):
            stage_cost += weight_ctrl * np.linalg.norm(u[:, i + 1] - u[:, i], 1)
            stage_cost += weight_xfg * np.linalg.norm(
                self.problem_parameters["goal_state"].value - self.variables["X"].value[0:6, i], 1
            )

        objective = terminal_cost + stage_cost
        obj = objective[0][0]
        return obj

    def _compute_s(self, x, p):
        s = np.zeros(self.params.K)
        for player_name in self.planets:
            x_cp = self.planets[player_name].center[0]
            y_cp = self.planets[player_name].center[1]
            r_p = self.planets[player_name].radius
            r_tot = r_p + self.r_buffer_stat_obs

            for k in range(self.params.K):
                value = -((x[0, k] - x_cp) ** 2 + (x[1, k] - y_cp) ** 2 - r_tot**2)
                if value >= 0:
                    s[k] += value
                else:
                    s[k] += 0
        return s

    def _compute_s_sat(self, x, p):
        s = np.zeros(self.params.K)
        index_satellite = 0
        for planet_name, satellite_params in self.satellites.items():
            tau = satellite_params.tau
            orbit_r = satellite_params.orbit_r
            omega = satellite_params.omega
            radius = satellite_params.radius
            # we need the relative planet's center
            planet_center = self.planets[planet_name.split("/")[0]].center
            planet_center = np.array(planet_center)

            for k in range(self.params.K):
                # Here we can avoid using the linearization (it's just a value)
                x_k = x[0, k]
                y_k = x[1, k]
                t_k = p * k / (self.params.K - 1)
                r = radius + self.r_buffer_stat_obs

                computation = -(
                    (x_k - planet_center[0] - orbit_r * np.cos(omega * t_k + tau)) ** 2
                    + (y_k - planet_center[1] - orbit_r * np.sin(omega * t_k + tau)) ** 2
                    - r**2
                )
                if computation > 0:
                    s[k] += computation
                else:
                    s[k] += 0
            index_satellite += 1
        return s

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
            "nu": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "nu_s": cvx.Variable(self.params.K),
            # "nu_ic": cvx.Variable(self.spaceship.n_x),
            "nu_tc": cvx.Variable(self.spaceship.n_x - 2),
            "nu_s_sat": cvx.Variable(self.params.K),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "goal_state": cvx.Parameter(self.spaceship.n_x - 2),
            # Linearization + discretization matrices
            "A_bar": [cvx.Parameter((self.spaceship.n_x, self.spaceship.n_x)) for k in range(self.params.K - 1)],
            "B_plus_bar": [cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u)) for k in range(self.params.K - 1)],
            "B_minus_bar": [cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u)) for k in range(self.params.K - 1)],
            "F_bar": [cvx.Parameter((self.spaceship.n_x, self.spaceship.n_p)) for k in range(self.params.K - 1)],
            "r_bar": [cvx.Parameter((self.spaceship.n_x)) for k in range(self.params.K - 1)],
            # Current guess
            "X_bar": cvx.Parameter((self.spaceship.n_x, self.params.K)),
            "U_bar": cvx.Parameter((self.spaceship.n_u, self.params.K)),
            "p_bar": cvx.Parameter(self.spaceship.n_p),
            # Trust region
            "tr_radius": cvx.Parameter(),
            # Docking
            "goal_minus_1": cvx.Parameter(3, name="goal_minus_1"),
            "goal_minus_2": cvx.Parameter(3, name="goal_minus_2"),
        }

        # Satellites
        if len(self.satellites) > 0:
            problem_parameters["C_sat"] = cvx.Parameter((2 * len(self.satellites), self.params.K), name="C_sat")
            problem_parameters["G_sat"] = cvx.Parameter((1 * len(self.satellites), self.params.K), name="G_sat")
            problem_parameters["r_prime_sat"] = cvx.Parameter(
                (1 * len(self.satellites), self.params.K), name="r_prime_sat"
            )

        return problem_parameters

    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        tf = self.variables["p"].value[0]
        timestep = tf / (self.params.K - 1)
        ts = tuple([i * timestep for i in range(self.params.K)])
        F = self.variables["U"].value[0, :]
        ddelta = self.variables["U"].value[1, :]
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        npstates = self.variables["X"].value.T
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates
