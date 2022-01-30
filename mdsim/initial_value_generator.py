"""
Module containing classes to generate initial values for the simulation.
"""

# Standard library
from typing import Union
from pathlib import Path

# 3rd-party packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import maxwell
import duq

# Self
from . import helpers
from .data.elements_data import data as elements_data


mpl.rcParams["figure.dpi"] = 100


class InitialValuesGenerator:
    """
    Superclass and general interface for all initial value generator classes.
    The parameters `unit_length` and `unit_time` are the absolute minimum needed
    to instantiate the class; depending on the specific initial-value generator, other parameters
    should be required in order to create the desired initial values.
    """

    def __init__(self, unit_length: Union[str, duq.Unit], unit_time: Union[str, duq.Unit]):

        # Convert given units to duq.Unit (if they are strings)
        # and verify that they have the correct dimension
        self._unit_length = helpers.convert_to_unit(unit_length, "L", "unit_length")
        self._unit_time = helpers.convert_to_unit(unit_time, "T", "unit_time")

        # Initialize attributes used to save the calculated initial values
        self._positions = None
        self._velocities = None
        self._atomic_nums = None
        self._mol_ids = None
        self._connectivity_matrix = None
        self._log = None

    @property
    def positions(self) -> np.ndarray:
        """
        Coordinates of all atoms in the system, as a 2D-array of shape (n, m), where 'n' is
        the number of all atoms, and 'm' is the number of spatial coordinates.
        """
        if self._positions is None:
            self._positions = self._calculate_positions()
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """
        Velocities of all atoms in the system, as a 2D-array of shape (n, m), where 'n' is
        the number of all atoms, and 'm' is the number of spatial coordinates.
        """
        if self._velocities is None:
            self._velocities = self._calculate_velocities()
        return self._velocities

    @property
    def atomic_numbers(self) -> np.ndarray:
        """
        Atomic numbers of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
        the number of all atoms.
        """
        if self._atomic_nums is None:
            self._atomic_nums = self._create_atomic_nums()
        return self._atomic_nums

    @property
    def molecule_ids(self) -> np.ndarray:
        """
        Molecule-IDs of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
        the number of all atoms.
        """
        if self._mol_ids is None:
            self._mol_ids = self._create_mol_ids()
        return self._mol_ids

    @property
    def connectivity_matrix(self) -> np.ndarray:
        """
        Indices of all bonded atoms to each atom, as a 1D-array of shape (n, ), where 'n' is the
        number of all atoms in the system. Each element of the array is then a list of indices.
        """
        if self._connectivity_matrix is None:
            self._connectivity_matrix = self._create_connectivity_matrix()
        return self._connectivity_matrix

    @property
    def unit_length(self) -> duq.Unit:
        """
        Length-unit of the values; the coordinates in `self.positions` are in this unit. Also, the
        unit of velocities in `self.velocities` is based on this unit (divided by `self.unit_time`)
        """
        return self._unit_length

    @property
    def unit_time(self) -> duq.Unit:
        """
        Time-unit of the values; the unit of velocities in `self.velocities` is based on this unit
        i.e. it is equal to `self.unit_length / self.unit_time`.
        """
        return self._unit_time

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
                unit_length=self.unit_length.exponents_as_is,
                unit_time=self.unit_time.exponents_as_is,
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

    def _create_atomic_nums(self) -> np.ndarray:
        """
        Create an array containing the atomic number of each atom in the system.

        Returns
        -------
        atomic_nums : numpy.ndarray
            Atomic numbers of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
            the number of all atoms.
        """
        pass

    def _create_mol_ids(self) -> np.ndarray:
        """
        Create a molecule-ID for each atom, specifying to which molecule it belongs.

        Returns
        -------
        mol_ids : numpy.ndarray
            Molecule-IDs of all atoms in the initial condition, as a 1D-array of shape (n, ), where
            'n' is the number of atoms in the system. Molecule-IDs can have arbitrary values, but
            should have the same value for all atoms in the same molecule.
        """
        pass

    def _create_connectivity_matrix(self) -> np.ndarray:
        """
        Create a list of indices for each atom, specifying the indices of all atoms, to which it
        is bonded.

        Returns
        -------
        bonded_atoms_idx : numpy.ndarray
            Indices of all bonded atoms to each atom, as a 1D-array of shape (n, ), where 'n' is
            the number of all atoms in the system. Each element of the array is then a list of
            indices.
        """
        pass

    def _create_log(self):
        pass


class Water(InitialValuesGenerator):
    """
    Initial-values generator for a collection of water molecules in a square or cubic shape,
    either in 2 or 3 spatial dimensions.

    Parameters
    ----------
    num_molecules_in_x : int

    num_molecules_in_y : int

    num_molecules_in_z : int

    temperature :

    pressure :

    eq_bond_len :

    eq_bond_angle :

    unit_length :

    unit_time :

    random_seed :

    """

    def __init__(
        self,
        num_molecules_in_x: int = 10,
        num_molecules_in_y: int = 10,
        num_molecules_in_z: int = 1,
        create_2d: bool = True,
        temperature: Union[str, duq.Quantity] = "310 K",
        pressure: Union[str, duq.Quantity] = "1 atm",
        eq_bond_len: Union[str, duq.Quantity] = "1.012 Å",
        eq_bond_angle: Union[str, duq.Quantity] = "113.24 deg",
        unit_length: Union[str, duq.Unit] = "Å",
        unit_time: Union[str, duq.Unit] = "fs",
        random_seed: Union[int, None] = None,
    ):

        super().__init__(unit_length, unit_time)

        # Verify that given number of molecules are all integers greater than 1
        helpers.raise_num_for_value_and_type(num_molecules_in_x, "num_molecules_in_x")
        helpers.raise_num_for_value_and_type(num_molecules_in_y, "num_molecules_in_y")
        helpers.raise_num_for_value_and_type(num_molecules_in_z, "num_molecules_in_z")
        # Assign to instance attributes
        self._num_mols_x = num_molecules_in_x
        self._num_mols_y = num_molecules_in_y
        self._num_mols_z = num_molecules_in_z
        # Calculate total number of molecules
        self._num_molecules_total = self._num_mols_x * self._num_mols_y * self._num_mols_z
        self._num_atoms_total = self._num_molecules_total * 3

        # Check whether the collection should be in 2D or 3D and verify if that's possible
        # considering the given number of molecules in different directions.
        if not isinstance(create_2d, bool):
            raise ValueError("Parameter `create_2d` should be boolean.")
        elif num_molecules_in_z != 1:
            if create_2d:
                raise ValueError(
                    "2D data can only be created when `num_molecules_in_z` is set to 1."
                )
            else:
                self._create_2d = False
        else:
            self._create_2d = create_2d

        # Convert given quantities to duq.Quantity objects (if they are strings)
        # and verify that they have the correct dimension
        self._temperature = helpers.convert_to_quantity(temperature, "Θ", "temperature")
        self._pressure = helpers.convert_to_quantity(pressure, "P", "pressure")
        self.eq_bond_length = helpers.convert_to_quantity(eq_bond_len, "L", "eq_bond_len")
        self.eq_bond_angle = helpers.convert_to_quantity(eq_bond_angle, "1", "eq_bond_angle")

        # Set up the random number generator
        if random_seed is not None:
            helpers.raise_num_for_value_and_type(random_seed, "random_seed")
        self._random_seed = random_seed
        self._random_gen = np.random.RandomState(self._random_seed)

        # Read masses and calculate molar mass of water
        self._mass_h = helpers.convert_to_quantity(elements_data[1]["mass"], "M", "mass_hydrogen")
        self._mass_o = helpers.convert_to_quantity(elements_data[8]["mass"], "M", "mass_oxygen")
        self._mass_molar_h2o = duq.Quantity(
            2 * self._mass_h.value + self._mass_o.value, "g.mol^-1"
        )

        # Initialize attributes needed to calculate positions
        self._density = None
        self._volume_per_molecule = None
        self._molecule_box_len = None
        self._box_coordinates = None

        return

    @property
    def density(self):
        if self._density is None:
            self._density = self._calculate_density()
        return self._density

    @property
    def volume_per_molecule(self):
        if self._volume_per_molecule is None:
            self._volume_per_molecule = self._calculate_volume_per_molecule()
        return self._volume_per_molecule

    @property
    def molecule_box_length(self):
        if self._molecule_box_len is None:
            self._molecule_box_len = self._calculate_molecule_box_length()
        return self._molecule_box_len

    @property
    def box_coordinates(self):
        if self._box_coordinates is None:
            self._box_coordinates = self._calculate_box_coordinates()
        return self._box_coordinates

    def _calculate_positions(self) -> np.ndarray:
        """
        Calculate the coordinates of each atom of each molecule in the water box, as an array
        of shape (n, m), where 'n' is the number of all atoms, and 'm' is the number of spatial
        dimensions (either 2 or 3, depending on whether `self._create_2D` is set to True or False,
        respectively). Atoms are ordered according to their molecules (i.e. the nth three atoms
        belong to the nth molecule), and for each molecule, the oxygen atom comes first.

        Returns
        -------
        positions : numpy.ndarray
        """
        q = np.zeros((self._num_atoms_total, 3))
        q_init = self._generate_single_molecule()

        for i in range(self._num_molecules_total):
            # generate a random angle (between 0 and 2π) for rotation in xy-plane
            random_angle = self._random_gen.random_sample(3) * 2 * np.pi
            # rotate the molecule in xy-plane (it will be rotated around its geometric
            # center, since the initially created molecule's geometric center is put on the origin)
            q_init_rot = helpers.rotate_3d_vector_around_axis(q_init, random_angle[0], "z")
            if not self._create_2d:
                q_init_rot = helpers.rotate_3d_vector(
                    q_init_rot, random_angle[1], random_angle[2], order="xy"
                )

            trans_unit_vector = np.array(
                [
                    i % self._num_mols_x,
                    (i // self._num_mols_x) % self._num_mols_y,
                    i // (self._num_mols_x * self._num_mols_y),
                ]
            )
            trans_vector = trans_unit_vector * self.molecule_box_length.value
            q_init_rot_trans = q_init_rot + trans_vector
            q[i * 3 : i * 3 + 3] = q_init_rot_trans

        # Center the box on the origin
        midpoint = (np.max(q, axis=0) + np.min(q, axis=0)) / 2
        q -= midpoint

        if self._create_2d:
            q = q[:, 0:2]
        return q

    def _calculate_molecule_box_length(self):
        return self.volume_per_molecule ** (1 / 3)

    def _calculate_box_coordinates(self) -> np.ndarray:
        num_mols = np.array([self._num_mols_x, self._num_mols_y, self._num_mols_z])
        box_half_lengths = num_mols * self.molecule_box_length.value / 2
        box_coordinates = np.array([-box_half_lengths, box_half_lengths])
        return box_coordinates

    def _calculate_velocities(self) -> np.ndarray:
        shape = (self._num_atoms_total, 2 if self._create_2d else 3)
        v = self._random_gen.random_sample(shape) * 2 - 1
        v_norm = v / np.linalg.norm(v, axis=1).reshape(-1, 1)

        temp = self._temperature.convert_unit("K")
        a_hydrogen = (duq.predefined_constants.boltzmann_const * temp / self._mass_h) ** (1 / 2)
        a_oxygen = (duq.predefined_constants.boltzmann_const * temp / self._mass_o) ** (1 / 2)
        unit_speed = self.unit_length / self.unit_time
        a_hydrogen_value = a_hydrogen.convert_unit(unit_speed).value
        a_oxygen_value = a_oxygen.convert_unit(unit_speed).value
        speeds = maxwell.rvs(
            size=(self._num_molecules_total, 3),
            scale=(a_oxygen_value, a_hydrogen_value, a_hydrogen_value),
            random_state=self._random_gen,
        ).reshape(-1, 1)
        velocities = v_norm * speeds
        return velocities

    def _create_atomic_nums(self) -> np.ndarray:
        atomic_nums = np.tile([8, 1, 1], self._num_molecules_total)
        return atomic_nums

    def _create_mol_ids(self) -> np.ndarray:
        mol_ids = np.repeat(np.arange(self._num_molecules_total), 3)
        return mol_ids

    def _create_connectivity_matrix(self) -> np.ndarray:
        matrix = np.full((self._num_atoms_total, self._num_atoms_total), False)
        for atom_idx in range(0, self._num_atoms_total, 3):
            matrix[atom_idx, atom_idx+1:atom_idx+3] = True
        return matrix

    def _generate_single_molecule(self) -> np.ndarray:
        """
        Generate the coordinates for atoms in a molecule of water,
        where the molecule is in the xy-plane, the geometric center is
        on the origin, the c2-rotation axis is the y-axis, and the
        hydrogen atoms have both the same distance from the oxygen,
        with negative y-values.

        Returns
        -------
        coordinates : numpy.ndarray
            2D-array of shape (3,3), where each element is the array of xyz-coordinates for one atom.
            The first element is the coordinates of oxygen, and the next two elements are of hydrogen atoms.
        """
        coordinates = np.zeros((3, 3))
        angle = self.eq_bond_angle.convert_unit("rad").value
        half_angle = angle / 2
        cos = np.cos(half_angle)
        sin = np.sin(half_angle)
        bond_length = self.eq_bond_length.convert_unit(self.unit_length).value
        x_hydrogens = bond_length * sin
        y_hydrogens = -bond_length * cos
        coordinates[1, :2] = [-x_hydrogens, y_hydrogens]
        coordinates[2, :2] = [x_hydrogens, y_hydrogens]
        geometric_center = (np.max(coordinates, axis=0) + np.min(coordinates, axis=0)) / 2
        centered_coordinates = coordinates - geometric_center
        return centered_coordinates

    def _calculate_volume_per_molecule(self) -> duq.Quantity:
        """
        Calculate the volume per molecule of water, based on the calculated density.

        Returns
        -------
            duq.Quantity
            Object representing the volume.
        """
        molar_density = self.density / self._mass_molar_h2o
        number_density = molar_density * duq.predefined_constants.avogadro_const
        particle_volume = 1 / number_density
        particle_volume.convert_unit(self.unit_length ** 3, inplace=True)
        return particle_volume

    def _calculate_density(self) -> duq.Quantity:
        """
        Calculate the density of water based on the given temperature and pressure.

        Returns
        -------
            duq.Quantity
            Object representing the density.
        """
        # TODO (calculate density as a function of temperature and pressure)
        density = duq.Quantity(997, "kg.m^-3")
        return density

    def _create_log(self):
        # TODO
        return

    def save_log_to_file(self, filepath):
        # TODO
        return

    def plot_ensemble(self):
        fig, ax = plt.subplots(figsize=(self._num_mols_x, self._num_mols_y))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
        ax.set_aspect("equal")
        plt.xlim(self.box_coordinates[0, 0], self.box_coordinates[1, 0])
        plt.ylim(self.box_coordinates[0, 1], self.box_coordinates[1, 1])
        mol_box_edges_x = np.array(
            [
                self.box_coordinates[0, 0] + i * self.molecule_box_length.value
                for i in range(self._num_mols_x + 1)
            ]
        )
        mol_box_edges_y = np.array(
            [
                self.box_coordinates[0, 1] + i * self.molecule_box_length.value
                for i in range(self._num_mols_y + 1)
            ]
        )
        plt.xticks(mol_box_edges_x, fontsize=8)
        plt.yticks(mol_box_edges_y, fontsize=8)

        plt.xlabel(f"x-coordinate [{self.unit_length.symbol_as_is}]")
        plt.ylabel(f"y-coordinate [{self.unit_length.symbol_as_is}]")

        # Plot molecule-box edges
        for mol_box_edge_x in mol_box_edges_x[1:-1]:
            plt.vlines(
                mol_box_edge_x, self.box_coordinates[0, 1], self.box_coordinates[1, 1], lw=0.5
            )
        for mol_box_edge_y in mol_box_edges_y:
            plt.hlines(
                mol_box_edge_y, self.box_coordinates[0, 0], self.box_coordinates[1, 0], lw=0.5
            )

        # ----- plot bonds as lines -----
        # Calculate line-width for the bond, depending on the size of figure
        # (formula fitted experimentally by trial and error to find the best coefficients)
        # TODO: define line-width in units of figure data (`lw` doesn't allow for this)
        bond_width = (
            max(self._num_mols_x, self._num_mols_y)
            / 10
            * 150
            / np.max(self.box_coordinates[1] - self.box_coordinates[0])
        )
        # Iterate over first hydrogen atoms
        for idx_atom in range(1, self._num_atoms_total, 3):
            # Draw a line from that hydrogen, to oxygen, to the other hydrogen
            ax.plot(
                [self.positions[idx_atom, 0], self.positions[idx_atom-1, 0], self.positions[idx_atom+1, 0]],
                [self.positions[idx_atom, 1], self.positions[idx_atom-1, 1], self.positions[idx_atom+1, 1]],
                lw=bond_width,
                color="black",
            )

        # plot atoms as circles
        for atom_idx in range(0, self._num_atoms_total, 3):
            oxygen = plt.Circle(
                (self.positions[atom_idx, 0], self.positions[atom_idx, 1]),
                float(elements_data[8]["cov_radius"].split()[0]),
                color=elements_data[8]["color"],
                zorder=1,
            )
            ax.add_patch(oxygen)
            hydrogen1 = plt.Circle(
                (self.positions[atom_idx+1, 0], self.positions[atom_idx+1, 1]),
                float(elements_data[1]["cov_radius"].split()[0]),
                color=elements_data[1]["color"],
                zorder=2,
            )
            ax.add_patch(hydrogen1)
            hydrogen2 = plt.Circle(
                (self.positions[atom_idx+2, 0], self.positions[atom_idx+2, 1]),
                float(elements_data[1]["cov_radius"].split()[0]),
                color=elements_data[1]["color"],
                zorder=2,
            )
            ax.add_patch(hydrogen2)
        plt.show()
        return
