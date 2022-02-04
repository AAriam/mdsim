"""
Module containing the superclass for all ensemble generator classes.
"""

# Standard library
from typing import Union
from pathlib import Path

# 3rd-party packages
import numpy as np
import duq

# Self
from .. import helpers


__all__ = ["EnsembleGenerator"]


class EnsembleGenerator:
    """
    Superclass for all initial value generator classes. The parameters `unit_length` and
    `unit_time` are the absolute minimum needed to instantiate the class; depending on the specific
    initial-value generator, other parameters should be required in order to create the desired
    initial values.
    """

    def __init__(
        self,
        temperature: Union[str, duq.Quantity] = "310 K",
        pressure: Union[str, duq.Quantity] = "1 atm",
        unit_length: Union[str, duq.Unit] = "Å",
        unit_time: Union[str, duq.Unit] = "fs",
        random_seed: Union[int, None] = None,
    ):
        """
        Create a simulation box.

        Parameters
        ----------
        unit_length : Union[str, duq.Unit]
            Unit of length used for generating the initial positions and velocities, either as a
            string representing the unit (e.g. "Å"), or as an equivalent `duq.Unit` object.
        unit_time : Union[str, duq.Unit]
            Unit of time used for generating the initial velocities, either as a string
            representing the unit (e.g. "fs"), or as an equivalent `duq.Unit` object.
        random_seed : int (optional; default: None)
            Seed for the random number generator used for applying random rotations to the
            particles, and drawing speeds from the Maxwell-Boltzmann distribution. If not specified
            , creating two ensembles with otherwise identical parameters will result in two
            different sets of initial values.
        """
        # Convert given quantities to duq.Quantity objects (if they are strings)
        # and verify that they have the correct dimension
        self._temperature = helpers.convert_to_quantity(temperature, "Θ", "temperature")
        self._pressure = helpers.convert_to_quantity(pressure, "P", "pressure")
        # Convert given units to duq.Unit (if they are strings)
        # and verify that they have the correct dimension
        self._unit_length = helpers.convert_to_unit(unit_length, "L", "unit_length")
        self._unit_time = helpers.convert_to_unit(unit_time, "T", "unit_time")
        # Derive the unit of velocity
        self._unit_velocity = self._unit_length / self._unit_time

        # Set up the random number generator
        if random_seed is not None:
            helpers.raise_num_for_value_and_type(random_seed, "random_seed")
        self._random_seed = random_seed
        self._random_gen = np.random.RandomState(self._random_seed)

        # Calculate metadata (should be implemented by subclasses)
        self._atomic_numbers = None
        self._molecule_ids = None
        self._connectivity_matrix = None
        # Calculate positions and velocities (should be implemented by subclasses)
        self._positions = None
        self._velocities = None
        self._log = None

    @property
    def positions(self) -> np.ndarray:
        """
        Coordinates of all atoms in the system, as a 2D-array of shape (n, m), where 'n' is
        the number of all atoms, and 'm' is the number of spatial coordinates.
        """
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """
        Velocities of all atoms in the system, as a 2D-array of shape (n, m), where 'n' is
        the number of all atoms, and 'm' is the number of spatial coordinates.
        """
        return self._velocities

    @property
    def atomic_numbers(self) -> np.ndarray:
        """
        Atomic numbers of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
        the number of all atoms.
        """
        return self._atomic_numbers

    @property
    def molecule_ids(self) -> np.ndarray:
        """
        Molecule-IDs of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
        the number of all atoms.
        """
        return self._molecule_ids

    @property
    def connectivity_matrix(self) -> np.ndarray:
        """
        Indices of all bonded atoms to each atom, as a 1D-array of shape (n, ), where 'n' is the
        number of all atoms in the system. Each element of the array is then a list of indices.
        """
        return self._connectivity_matrix

    @property
    def temperature(self) -> duq.Quantity:
        """
        Temperature of the system.

        Returns
        -------
        temperature : duq.Quantity
            Temperature as a `duq.Quantity` object.
        """
        return self._temperature

    @property
    def pressure(self) -> duq.Quantity:
        """
        Pressure of the system.

        Returns
        -------
        pressure : duq.Quantity
            Pressure as a `duq.Quantity` object.
        """
        return self._pressure

    @property
    def unit_positions(self) -> duq.Unit:
        """
        Unit of the positions of atoms.
        """
        return self._unit_length

    @property
    def unit_velocities(self) -> duq.Unit:
        """
        Unit of the velocities of atoms.
        """
        return self._unit_velocity

    @property
    def log(self):
        if self._log is None:
            self._create_log()
        return self._log

    def save_to_file(self, filepath: Union[str, Path]) -> Path:
        """
        Save all data on the calculated initial values to a file (numpy uncompressed .npz format).

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Local path to save the file; specifying the file extension is not necessary.

        Returns
        -------
            pathlib.Path
            Full path of the saved file.
        """
        fullpath = Path(filepath).with_suffix(".npz")
        with open(fullpath, "wb+") as f:
            np.savez(
                f,
                positions=self.positions,
                velocities=self.velocities,
                atomic_numbers=self.atomic_numbers,
                molecule_ids=self.molecule_ids,
                connectivity_matrix=self.connectivity_matrix,
                unit_positions=self.unit_positions.exponents_as_is,
                unit_velocities=self.unit_velocities.exponents_as_is,
            )
        return fullpath.absolute()

    def _calculate_positions(self) -> np.ndarray:
        """
        Calculate the position vector of each atom in the system.

        Returns
        -------
        positions : numpy.ndarray
            Coordinates of all atoms in the initial condition, as a 2D-array of shape (n, m), where
            'n' is the number of atoms in the system, and 'm' is the number of spatial coordinates.
        """
        pass

    def _calculate_velocities(self) -> np.ndarray:
        """
        Calculate the velocity vector of each atom in the system.

        Returns
        -------
        velocities : numpy.ndarray
            Velocities of all atoms in the initial condition, as a 2D-array of shape (n, m), where
            'n' is the number of atoms in the system, and 'm' is the number of spatial coordinates.
        """
        pass

    def _calculate_atomic_numbers(self) -> np.ndarray:
        """
        Create an array containing the atomic number of each atom in the system.

        Returns
        -------
        atomic_nums : numpy.ndarray
            Atomic numbers of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
            the number of all atoms.
        """
        pass

    def _calculate_molecule_ids(self) -> np.ndarray:
        """
        Create a molecule-ID for each atom, specifying to which molecule it belongs.

        Returns
        -------
        molecule_ids : numpy.ndarray
            Molecule-IDs of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is the
            number of atoms in the system. Molecule-IDs can have arbitrary values, but should have
            the same value for all atoms in the same molecule.
        """
        pass

    def _calculate_connectivity_matrix(self) -> np.ndarray:
        """
        Create a connectivity matrix for all atoms in the system, specifying the connectivity of
        each atom to other atoms.

        Returns
        -------
        connectivity_matrix : numpy.ndarray
            Matrix of shape (n, n), where 'n' is the number of atoms in the system. Each matrix
            element 'c_ij' is a boolean specifying whether the atom at index 'i' is connected to
            the atom at index 'j'. For each pair of atoms, this data is only set at the matrix
            element 'c_ij' where 'i' is smaller than 'j' (i.e. only the upper triangle of the
            matrix contains the data).
        """
        pass

    def _create_log(self):
        pass
