"""
Module containing the class MDSimulation for running an MD simulation.
"""

# Standard library
from pathlib import Path

# 3rd-party packages
import numpy as np
import numintegrator as ode
import duq
from mdforce.models.forcefield_superclass import ForceField

# Self
from . import initial_value_generator as init_gen
from .ensemble_generator.superclass import EnsembleGenerator


__all__ = ["MDSimulation"]


class MDSimulation:
    def __init__(self, forcefield: ForceField, ensemble: EnsembleGenerator):
        """
        Initialize an MD-simulation on a given initial ensemble using a given force-field.

        Parameters
        ----------
        forcefield : mdforce.models.forcefield_superclass.ForceField
            Force-field to use in the simulation.
        ensemble : mdsim.ensemble_generator.superclass.EnsembleGenerator
            Initial ensemble to run the simulation on.
        """
        # Verify type of input arguments and assign as instance attributes
        if not isinstance(forcefield, ForceField) or (
            not issubclass(forcefield.__class__, ForceField)
        ):
            raise ValueError(
                "Argument `forcefield` should either be an instance or a subclass of "
                "`mdforce.models.forcefield_superclass.ForceField`."
            )
        else:
            self._forcefield = forcefield
        if not isinstance(ensemble, EnsembleGenerator) or (
            not issubclass(ensemble.__class__, EnsembleGenerator)
        ):
            raise ValueError(
                "Argument `ensemble` should either be an instance or a subclass of "
                "`mdsim.ensemble_generator.superclass.EnsembleGenerator`."
            )
        else:
            self._ensemble = ensemble
        # Initialize instance attributes for storing the simulation results
        self._trajectory: TrajectoryAnalyzer = None
        self._positions: np.ndarray = None
        self._velocities: np.ndarray = None
        self._timestamps: np.ndarray = None
        self._energy_potential_coulomb: np.ndarray = None
        self._energy_potential_lennard_jones: np.ndarray = None
        self._energy_potential_bond_vibration: np.ndarray = None
        self._energy_potential_angle_vibration: np.ndarray = None
        self._bond_angles: np.ndarray = None
        self._distances_interatomic: np.ndarray = None
        self._curr_step: int = None
        return

    @property
    def trajectory(self) -> TrajectoryAnalyzer:
        """
        TrajectoryAnalyzer object containing all the simulation data, in addition to
        functionalities for calculating new properties and visualization.

        Returns
        -------
        trajectory : TrajectoryAnalyzer
        """
        if self._trajectory is None:
            raise ValueError("The simulation has not yet been run.")
        else:
            return self._trajectory

    @property
    def ensemble(self) -> EnsembleGenerator:
        """
        EnsembleGenerator object containing the initial values for the simulation.

        Returns
        -------
        ensemble: EnsembleGenerator
        """
        return self._ensemble

    @property
    def forcefield(self) -> ForceField:
        """
        ForceField object containing the force-field used in the simulation.

        Returns
        -------
        forcefield : ForceField
        """
        return self._forcefield

    def run(
        self,
        num_steps: int = 1000,
        dt: float = 1,
        pbc: bool = False,
    ):
        """
        Run the MD-simulation on the ensemble for a given number of steps and step-size.

        Parameters
        ----------
        num_steps : int
            Number of integration steps.
        dt : float
            Step-size in the time-unit of ensemble.
        pbc : bool
            Whether to run the simulation according to periodic boundary conditions.

        Returns
        -------
            None
            All the simulation data are stored in a TrajectoryAnalyzer object accessible at
            `self.trajectory`.
        """
        # Create arrays for storing force-field calculations
        self._energy_potential_coulomb = np.zeros(num_steps + 1)
        self._energy_potential_lennard_jones = np.zeros(num_steps + 1)
        self._energy_potential_bond_vibration = np.zeros(num_steps + 1)
        self._energy_potential_angle_vibration = np.zeros(num_steps + 1)
        self._bond_angles = np.zeros((num_steps + 1, self._num_molecules))
        self._distances_interatomic = np.zeros((num_steps + 1, self._num_atoms, self._num_atoms))

        # Run integration
        self._positions, self._velocities, self._timestamps = ode.integrate(
            integrator=ode.Integrators.ODE_2_EXPLICIT_VELOCITY_VERLET,
            f=self._force,
            x0=self._init_positions,
            v0=self._init_velocities,
            dt=dt,
            n_steps=num_steps,
        )
        pass

    def _force(self, q: np.ndarray, t=None):
        """
        Force-function to pass to the integrator; since the integrator expects a function with two
        arguments, another dummy argument `t` is added. In each integration step, this function
        takes the positions and passes them to the force-field; it then extracts the calculated
        data by the force-field and stores them in instance attributes, and returns the calculated
        acceleration back to the integrator.

        Parameters
        ----------
        q : numpy.ndarray
            Positions of all atoms in the current step.
        t : None
            Dummy argument, since the integrator expects a function with two arguments.

        Returns
        -------
        acceleration : numpy.ndarray
            Acceleration vector for each atom in the current step.
        """
        # Update force-field
        self._forcefield(q)

        # Extract calculated data from force-field
        self._energy_potential_coulomb[self._curr_step] = self._forcefield.energy_coulomb
        self._energy_potential_lennard_jones[
            self._curr_step
        ] = self._forcefield.energy_lennard_jones
        self._energy_potential_bond_vibration[
            self._curr_step
        ] = self._forcefield.energy_bond_vibration
        self._energy_potential_angle_vibration[
            self._curr_step
        ] = self._forcefield.energy_angle_vibration
        self._bond_angles[self._curr_step, ...] = self._forcefield.bond_angles
        self._distances_interatomic[self._curr_step, ...] = self._forcefield.distances

        # Increment the current step
        self._curr_step += 1
        # Return acceleration
        return self._forcefield.acceleration
