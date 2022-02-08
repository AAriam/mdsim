"""
Main module for running an MD simulation.
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

    @staticmethod
    def _raise_for_none(attr, sim_or_data="sim"):
        if attr is None:
            if sim_or_data == "sim":
                raise ValueError("The simulation has not yet been run.")
            else:
                raise ValueError("The initial values have not yet been loaded.")
        else:
            return attr
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

        self._curr_step += 1
        return self._forcefield.acceleration

    def _derive_num_atoms_and_molecules(self):
        self._num_atoms = self._atomic_numbers.size
        self._num_molecules = np.unique(self._molecule_ids).size
        self._num_spatial_dims = self._init_positions.shape[1]
        return

