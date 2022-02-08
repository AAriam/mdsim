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


class MDSimulation:
    def __init__(
        self,
        forcefield: ForceField,
    ):

        self._forcefield = forcefield

        self._init_positions = None
        self._init_velocities = None
        self._init_unit_length = None
        self._init_unit_time = None
        self._atomic_numbers = None
        self._molecule_ids = None
        self._connectivity_matrix = None
        self._num_atoms = None
        self._num_molecules = None

        self._positions = None
        self._velocities = None
        self._timestamps = None

        self._energy_potential_coulomb = None
        self._energy_potential_lennard_jones = None
        self._energy_potential_bond_vibration = None
        self._energy_potential_angle_vibration = None
        self._bond_angles = None
        self._distances_interatomic = None

        self._curr_step = 0

    @property
    def positions(self):
        return self._raise_for_none(self._positions)

    @property
    def velocities(self):
        return self._raise_for_none(self._velocities)

    @property
    def timestamps(self):
        return self._raise_for_none(self._timestamps)

    @property
    def energy_potential_coulomb(self):
        return self._raise_for_none(self._energy_potential_coulomb)

    @property
    def energy_potential_lennard_jones(self):
        return self._raise_for_none(self._energy_potential_lennard_jones)

    @property
    def energy_potential_bond_vibration(self):
        return self._raise_for_none(self._energy_potential_bond_vibration)

    @property
    def energy_potential_angle_vibration(self):
        return self._raise_for_none(self._energy_potential_angle_vibration)

    @property
    def bond_angles(self):
        return self._raise_for_none(self._bond_angles)

    @property
    def distances_interatomic(self):
        return self._raise_for_none(self._distances_interatomic)

    @property
    def atomic_numbers(self):
        return self._raise_for_none(self._atomic_numbers, "data")

    @property
    def molecule_ids(self):
        return self._raise_for_none(self._molecule_ids, "data")

    @property
    def bonded_atoms_indices(self):
        return self._raise_for_none(self._connectivity_matrix, "data")

    @property
    def masses(self):
        return self._raise_for_none(self._masses, "data")

    @staticmethod
    def _raise_for_none(attr, sim_or_data="sim"):
        if attr is None:
            if sim_or_data == "sim":
                raise ValueError("The simulation has not yet been run.")
            else:
                raise ValueError("The initial values have not yet been loaded.")
        else:
            return attr

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

    def _force(self, q, t):
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

    def load_initial_values_from_generator(self, generator: init_gen.InitialValuesGenerator):
        if not issubclass(generator.__class__, init_gen.InitialValuesGenerator):
            raise ValueError(
                "`generator` should be a subclass of "
                "mdsim.initial_value_generator.InitialValuesGenerator"
            )
        else:
            self._init_positions = generator.positions
            self._init_velocities = generator.velocities
            self._init_unit_length = generator.unit_length
            self._init_unit_time = generator.unit_time
            self._atomic_numbers = generator.atomic_numbers
            self._molecule_ids = generator.molecule_ids
            self._connectivity_matrix = generator.connectivity_matrix

            self._derive_num_atoms_and_molecules()
            self._forcefield.initialize_forcefield(self._init_positions.shape)
            self._forcefield.fit_units_to_input_data(
                self._init_unit_length,
                self._init_unit_time,
            )

            self._masses = np.tile(
                [generator._mass_o.value, generator._mass_h.value, generator._mass_h.value],
                self._num_molecules,
            )
        return

    def _derive_num_atoms_and_molecules(self):
        self._num_atoms = self._atomic_numbers.size
        self._num_molecules = np.unique(self._molecule_ids).size
        self._num_spatial_dims = self._init_positions.shape[1]
        return

    def save_all_data_to_file(self, filepath):
        path = Path(filepath)
        with open(path.with_suffix("npz"), "wb+") as f:
            np.savez(
                f,
                positions=self.positions,
                velocity=self.velocities,
                distances=self.distances_interatomic,
                angles_bond=self.bond_angles,
                energy_potential_coulomb=self.energy_potential_coulomb,
                energy_potential_lennard_jones=self.energy_potential_lennard_jones,
                energy_potential_bond_vibration=self.energy_potential_bond_vibration,
                energy_potential_angle_vibration=self.energy_potential_angle_vibration,
            )
        return

    def save_trajectory_to_file(self, filepath):
        path = Path(filepath)
        with open(path.with_suffix("npz"), "wb+") as f:
            np.savez(f, positions=self.positions, velocity=self.velocities)
        return

    def save_secondary_data_to_file(self, filepath):
        path = Path(filepath)
        with open(path.with_suffix("npz"), "wb+") as f:
            np.savez(
                f,
                distances=self.distances_interatomic,
                angles_bond=self.bond_angles,
                energy_potential_coulomb=self.energy_potential_coulomb,
                energy_potential_lennard_jones=self.energy_potential_lennard_jones,
                energy_potential_bond_vibration=self.energy_potential_bond_vibration,
                energy_potential_angle_vibration=self.energy_potential_angle_vibration,
            )
        return

    def save_metadata_to_file(self, filepath):
        pass
