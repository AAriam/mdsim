"""
Module containing the superclass for all ensemble-generator classes.
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
    Superclass for all ensemble-generator classes. The parameters, attributes and methods of this
    class are the absolute minimum shared between all ensemble-generator subclasses. Depending on
    the specific generator, other parameters, attributes and methods should be implemented in order
    to create the desired ensemble.
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
        temperature : Union[str, duq.Quantity]
            Desired temperature of the system, either as a string representing the value and the
            unit (e.g. "310 K"), or as an equivalent `duq.Quantity` object. The temperature is used
            to assign initial velocities to the atoms.
        pressure : Union[str, duq.Quantity]
            Desired pressure of the system, either as a string representing the value and the
            unit (e.g. "1 atm"), or as an equivalent `duq.Quantity` object. The pressure, along
             with the temperature, is used to calculate the density of the system.
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
        self._box_coordinates = None
        self._box_lengths = None
        self._num_molecules_total = None
        self._num_atoms_total = None
        self._is_2d = None
        # Calculate positions and velocities (should be implemented by subclasses)
        self._positions = None
        self._velocities = None
        self._log = None

    @property
    def positions(self) -> np.ndarray:
        """
        Coordinate vector of each atom in the system.

        Returns
        -------
        positions : numpy.ndarray
            2D-array of shape (n, m), where 'n' is the total number of atoms in the system, and 'm'
            is the number of spatial coordinates (i.e. either 2 or 3).
        """
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """
        Velocity vector of each atom in the system.

        Returns
        -------
        velocities : numpy.ndarray
            2D-array of shape (n, m), where 'n' is the total number of atoms in the system, and 'm'
            is the number of spatial coordinates (i.e. either 2 or 3).
        """
        return self._velocities

    @property
    def box_coordinates(self) -> np.ndarray:
        """
        Coordinates of the two opposite edges of the simulation box.

        Returns
        -------
        box_coordinates : numpy.ndarray
            2D-array of shape (2, m), where 'm' is either 2 or 3, depending on whether the ensemble
            is in 2D or 3D. The first element of the array corresponds to the minimum edge of the
            box, whereas the second element corresponds to the maximum edge.
        """
        return self._box_coordinates

    @property
    def box_lengths(self) -> np.ndarray:
        """
        Lengths of the simulation box.

        Returns
        -------
        box_lengths : numpy.ndarray
            1D-array of shape (m, ), where 'm' is either 2 or 3, depending on whether the ensemble
            is in 2D or 3D. The lengths are listed in the order x, y, z.
        """
        return self._box_lengths

    @property
    def atomic_numbers(self) -> np.ndarray:
        """
        Atomic number of each atom in the system.

        Returns
        -------
        atomic_numbers : numpy.ndarray
            1D-array of shape (n, ), where 'n' is the total number of atoms in the system.
        """
        return self._atomic_numbers

    @property
    def molecule_ids(self) -> np.ndarray:
        """
        Molecule-ID of each atom in the system, specifying to which molecule it belongs.

        Returns
        -------
        molecule_ids : numpy.ndarray
            1D-array of shape (n, ), where 'n' is the total number of atoms in the system.
        """
        return self._molecule_ids

    @property
    def connectivity_matrix(self) -> np.ndarray:
        """
        Connectivity matrix of the atoms in the system, specifying the connectivity of each atom to
        other atoms.

        Returns
        -------
        connectivity_matrix : numpy.ndarray
            Matrix of shape (n, n), where 'n' is the number of atoms in the system. Each matrix
            element 'c_ij' is a boolean specifying whether the atom at index 'i' is connected to
            the atom at index 'j'. For each pair of atoms, this data is only set at the matrix
            element 'c_ij' where 'i' is smaller than 'j' (i.e. only the upper triangle of the
            matrix contains the data).
        """
        return self._connectivity_matrix

    @property
    def number_molecules_total(self) -> int:
        """
        Total number of molecules in the system.

        Returns
        -------
        num_molecules_total : int
        """
        return self._num_molecules_total

    @property
    def number_atoms_total(self) -> int:
        """
        Total number of atoms in the system.

        Returns
        -------
        num_atoms_total : int
        """
        return self._num_atoms_total

    @property
    def number_spatial_dimensions(self) -> int:
        """
        Number of spatial dimensions for the data; i.e. 2 if the ensemble is 2D, and 3 if it's 3D.

        Returns
        -------
        number_spatial_dimensions : int
        """
        return 2 if self._is_2d else 3

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

        Returns
        -------
        unit_positions : duq.Unit
            Unit as a `duq.Unit` object.
        """
        return self._unit_length

    @property
    def unit_time(self) -> duq.Unit:
        """
        Unit of time used in velocities.

        Returns
        -------
        unit_positions : duq.Unit
            Unit as a `duq.Unit` object.
        """
        return self._unit_time

    @property
    def unit_velocities(self) -> duq.Unit:
        """
        Unit of the velocities of atoms.

        Returns
        -------
        unit_velocities : duq.Unit
            Unit as a `duq.Unit` object.
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
                box_coordinates=self._box_coordinates,
                atomic_numbers=self.atomic_numbers,
                molecule_ids=self.molecule_ids,
                connectivity_matrix=self.connectivity_matrix,
                unit_length=self.unit_positions.exponents_as_is,
                unit_time=self.unit_time.exponents_as_is,
                unit_temperature=self.temperature.unit.exponents_as_is,
                unit_pressure=self.pressure.unit.exponents_as_is,
                temperature_pressure=np.array([self.temperature.value, self.pressure.value]),
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
