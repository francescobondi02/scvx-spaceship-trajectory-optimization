from dataclasses import dataclass

from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


from pdm4ar.exercises.ex11.planner import SpaceshipPlanner
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString
from pathlib import Path


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    X: np.ndarray
    U: np.ndarray
    p: np.ndarray

    def __init__(
        self,
        init_state: SpaceshipState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params

        self.static_obstacles = init_sim_obs.dg_scenario.static_obstacles
        self.box = None

        for obstacle in self.static_obstacles:
            if isinstance(obstacle.shape, LineString):
                self.box = obstacle
                break

        self.planner = SpaceshipPlanner(
            planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp, box=self.box
        )
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.goal_state = init_sim_obs.goal.target
        if isinstance(init_sim_obs.goal, DockingTarget):
            print("Docking Target")

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, init_sim_obs.goal)

        # Plot
        # self.plot_states(self.state_traj._values)
        # self.plot_commands(self.cmds_plan._values)

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)


        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        expected_state = self.state_traj.at_interp(sim_obs.time)

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold

        cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds

    def plot_states(self, state_values: tuple[SpaceshipState]):
        x_coords = [state.x for state in state_values]
        y_coords = [state.y for state in state_values]
        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, y_coords, marker="o", linestyle="-", label="Trajectory")
        plt.quiver(
            x_coords,
            y_coords,
            np.cos([state.psi for state in state_values]),
            np.sin([state.psi for state in state_values]),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
            label="Direction",
        )
        plt.quiver(
            x_coords,
            y_coords,
            np.cos([state.delta + state.psi + np.pi for state in state_values]),
            np.sin([state.delta + state.psi + np.pi for state in state_values]),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            label="Thrust_Angle",
        )
        plt.title("Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Check if already saved (WARNING remember to remove the previous images)
        f_path_1 = Path("debug/computed_trajectory_1.png")
        f_path_2 = Path("debug/computed_trajectory_2.png")
        f_path_3 = Path("debug/computed_trajectory_3.png")
        f_path_4 = Path("debug/computed_trajectory_4.png")

        # Controllo e rimozione dei file # TODO to be tested
        if f_path_1.exists() and f_path_2.exists() and f_path_3.exists() and f_path_4.exists():
            f_path_1.unlink()  # Elimina il file
            f_path_2.unlink()
            f_path_3.unlink()
            f_path_4.unlink()
            print("Tutti i file sono stati eliminati.")
        else:
            print("Non tutti i file sono presenti. Nessun file eliminato.")

        if f_path_1.exists() and f_path_2.exists() and f_path_3.exists():
            plt.savefig("debug/computed_trajectory_4.png")
        elif f_path_1.exists() and f_path_2.exists():
            plt.savefig("debug/computed_trajectory_3.png")
        elif f_path_1.exists():
            plt.savefig("debug/computed_trajectory_2.png")
        else:
            plt.savefig("debug/computed_trajectory_1.png")

    def plot_commands(self, values: tuple[SpaceshipCommands]):
        thrust = [value.thrust for value in values]
        delta = [value.ddelta for value in values]
        plt.figure(figsize=(10, 6))
        plt.plot(thrust, marker="o", linestyle="-", label="Thrust", color="red")
        plt.plot(delta, marker="o", linestyle="-", label="Delta", color="blue")
        plt.title("Commands")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.show()

        f_path_1 = Path("debug/computed_commands_1.png")
        f_path_2 = Path("debug/computed_commands_2.png")
        f_path_3 = Path("debug/computed_commands_3.png")
        f_path_4 = Path("debug/computed_commands_4.png")

        # Controllo e rimozione dei file # TODO to be tested
        if f_path_1.exists() and f_path_2.exists() and f_path_3.exists() and f_path_4.exists():
            f_path_1.unlink()  # Elimina il file
            f_path_2.unlink()
            f_path_3.unlink()
            f_path_4.unlink()
            print("Tutti i file sono stati eliminati.")
        else:
            print("Non tutti i file sono presenti. Nessun file eliminato.")

        if f_path_1.exists() and f_path_2.exists() and f_path_3.exists():
            plt.savefig("debug/computed_commands_4.png")
        elif f_path_1.exists() and f_path_2.exists():
            plt.savefig("debug/computed_commands_3.png")
        elif f_path_1.exists():
            plt.savefig("debug/computed_commands_2.png")
        else:
            plt.savefig("debug/computed_commands_1.png")
