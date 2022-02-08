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
from .superclass import EnsembleGenerator
from .. import helpers
from ..data.elements_data import data as elements_data


__all__ = ["Water"]


class Water(EnsembleGenerator):
    """
    Initial-values generator for a collection of water molecules, either in 2 or 3 spatial
    dimensions.
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
        """
        Create an ensemble of water molecules.

        Parameters
        ----------
        num_molecules_in_x : int
            Number of molecules along the x-axis.
        num_molecules_in_y : int
            Number of molecules along the y-axis.
        num_molecules_in_z : int
            Number of molecules along the z-axis. This should be set to 1 when a 2D-system is
            desired.
        create_2d : bool
            Whether to create a 2D-system; only possible when `num_molecules_in_z` is set to 1.
            If `num_molecules_in_z` is set to 1 but `create_2d` is False, a sheet of water
            molecules is created, where each molecule is also rotated along the x- and y-axes (and
            thus each atom has a non-zero z-coordinate as well).
        temperature : Union[str, duq.Quantity]
            Desired temperature of the system, either as a string representing the value and the
            unit (e.g. "310 K"), or as an equivalent `duq.Quantity` object. The temperature is used
            to assign initial velocities to the atoms.
        pressure : Union[str, duq.Quantity]
            Desired pressure of the system, either as a string representing the value and the
            unit (e.g. "1 atm"), or as an equivalent `duq.Quantity` object. The pressure, along
             with the temperature, is used to calculate the density of the system.
        eq_bond_len : Union[str, duq.Quantity]
            Equilibrium bond length for the O–H bond; all bonds in the system will have this
            length.
        eq_bond_angle : Union[str, duq.Quantity]
            Equilibrium bond angle for the H–O–H angle; all water molecules in the system will have
            this angle.
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
        # Instantiate the superclass
        super().__init__(temperature, pressure, unit_length, unit_time, random_seed)
        # Verify that given number of molecules are all integers greater than 1
        helpers.raise_num_for_value_and_type(num_molecules_in_x, "num_molecules_in_x")
        helpers.raise_num_for_value_and_type(num_molecules_in_y, "num_molecules_in_y")
        helpers.raise_num_for_value_and_type(num_molecules_in_z, "num_molecules_in_z")
        # Assign to instance attributes
        self._num_mols_x = num_molecules_in_x
        self._num_mols_y = num_molecules_in_y
        self._num_mols_z = num_molecules_in_z
        # Calculate total number of molecules and atoms
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
                self._is_2d = False
        else:
            self._is_2d = create_2d
        # Convert given quantities to duq.Quantity objects (if they are strings)
        # and verify that they have the correct dimension
        self._eq_bond_length = helpers.convert_to_quantity(eq_bond_len, "L", "eq_bond_len")
        self._eq_bond_angle = helpers.convert_to_quantity(eq_bond_angle, "1", "eq_bond_angle")
        # Read masses and calculate molar mass of water
        self._mass_h = duq.Quantity(elements_data[1]["mass_Da"], "Da")
        self._mass_o = duq.Quantity(elements_data[8]["mass_Da"], "Da")
        self._mass_molar_h2o = duq.Quantity(
            2 * self._mass_h.value + self._mass_o.value, "g.mol^-1"
        )
        # Initialize attributes needed to calculate positions

        self._atomic_numbers = self._calculate_atomic_numbers()
        self._molecule_ids = self._calculate_molecule_ids()
        self._connectivity_matrix = self._calculate_connectivity_matrix()

        self._velocities = self._calculate_velocities()

        self._density = self._calculate_density()
        self._volume_per_molecule = self._calculate_volume_per_molecule()
        self._molecule_box_length = self._calculate_molecule_box_length()
        self._box_coordinates = self._calculate_box_coordinates()
        self._box_lengths = self._calculate_box_lengths()

        self._positions = self._calculate_positions()
        return

    @property
    def density(self) -> duq.Quantity:
        """
        Density of the system, calculated from the given temperature and pressure.

        Returns
        -------
        density : duq.Quantity
        """
        return self._density

    @property
    def volume_per_molecule(self):
        return self._volume_per_molecule

    @property
    def molecule_box_length(self):
        return self._molecule_box_length

    def plot_ensemble(self, figure_dpi=100):
        mpl.rcParams["figure.dpi"] = figure_dpi
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

        plt.xlabel(f"x-coordinate [{self.unit_positions.symbol_as_is}]")
        plt.ylabel(f"y-coordinate [{self.unit_positions.symbol_as_is}]")

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
                [
                    self.positions[idx_atom, 0],
                    self.positions[idx_atom - 1, 0],
                    self.positions[idx_atom + 1, 0],
                ],
                [
                    self.positions[idx_atom, 1],
                    self.positions[idx_atom - 1, 1],
                    self.positions[idx_atom + 1, 1],
                ],
                lw=bond_width,
                color="black",
            )

        # plot atoms as circles
        for atom_idx in range(0, self._num_atoms_total, 3):
            oxygen = plt.Circle(
                (self.positions[atom_idx, 0], self.positions[atom_idx, 1]),
                float(elements_data[8]["cov_radius"].split()[0]),
                color=elements_data[8]["color"],
                zorder=10,
            )
            ax.add_patch(oxygen)
            hydrogen1 = plt.Circle(
                (self.positions[atom_idx + 1, 0], self.positions[atom_idx + 1, 1]),
                float(elements_data[1]["cov_radius"].split()[0]),
                color=elements_data[1]["color"],
                zorder=20,
            )
            ax.add_patch(hydrogen1)
            hydrogen2 = plt.Circle(
                (self.positions[atom_idx + 2, 0], self.positions[atom_idx + 2, 1]),
                float(elements_data[1]["cov_radius"].split()[0]),
                color=elements_data[1]["color"],
                zorder=20,
            )
            ax.add_patch(hydrogen2)
        plt.show()
        return

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
        # Create an empty array to store the positions
        q = np.zeros((self._num_atoms_total, 3))
        # Create a single molecule as reference point
        q_init = self._generate_single_molecule()
        # Iterate over the total number of molecules
        for i in range(self._num_molecules_total):
            # Generate a random angle (between 0 and 2π) for rotation in xy-plane
            random_angle = self._random_gen.random_sample(3) * 2 * np.pi
            # Rotate the molecule in xy-plane (it will be rotated around its geometric
            # center, since the initially created molecule's geometric center is put on the origin)
            q_init_rot = helpers.rotate_3d_vector_around_axis(q_init, random_angle[0], "z")
            # If the simulation box is to be in 3D, also rotate the molecule about the x- and
            # y-axes
            if not self._is_2d:
                q_init_rot = helpers.rotate_3d_vector(
                    q_init_rot, random_angle[1], random_angle[2], order="xy"
                )
            # Create a translation unit vector, pointing from the origin to the current cell
            trans_unit_vector = np.array(
                [
                    i % self._num_mols_x,
                    (i // self._num_mols_x) % self._num_mols_y,
                    i // (self._num_mols_x * self._num_mols_y),
                ]
            )
            # Multiply the unit vector with the box-length of a single molecule, to get the
            # translation vector in units of length
            trans_vector = trans_unit_vector * self.molecule_box_length.value
            # Shift the coordinates of the current molecule into its current cell
            q_init_rot_trans = q_init_rot + trans_vector
            q[i * 3 : i * 3 + 3] = q_init_rot_trans
        # Center the box on the origin
        midpoint = (np.max(q, axis=0) + np.min(q, axis=0)) / 2
        q -= midpoint
        # If the simulation box is 2D, remove (all zero) z-coordinates
        if self._is_2d:
            q = q[:, 0:2]
        return q

    def _calculate_velocities(self) -> np.ndarray:
        # Derive the shape of data
        shape = (self._num_atoms_total, 2 if self._is_2d else 3)
        # Create a random unit vector for each atom
        v = self._random_gen.random_sample(shape) * 2 - 1
        v_norm = v / np.linalg.norm(v, axis=1).reshape(-1, 1)
        # Calculate the parameter of the Maxwell distribution function in scipy
        temp = self._temperature.convert_unit("K")
        a_hydrogen = (duq.predefined_constants.boltzmann_const * temp / self._mass_h) ** (1 / 2)
        a_oxygen = (duq.predefined_constants.boltzmann_const * temp / self._mass_o) ** (1 / 2)
        a_hydrogen_value = a_hydrogen.convert_unit(self.unit_velocities).value
        a_oxygen_value = a_oxygen.convert_unit(self.unit_velocities).value
        # Draw speeds from the Maxwell-Boltzmann distribution
        speeds = maxwell.rvs(
            size=(self._num_molecules_total, 3),
            scale=(a_oxygen_value, a_hydrogen_value, a_hydrogen_value),
            random_state=self._random_gen,
        ).reshape(-1, 1)
        # Multiply speeds with unit vectors to get velocity vectors for each atom
        velocities = v_norm * speeds
        return velocities

    def _calculate_box_coordinates(self) -> np.ndarray:
        if self._is_2d:
            num_mols = np.array([self._num_mols_x, self._num_mols_y])
        else:
            num_mols = np.array([self._num_mols_x, self._num_mols_y, self._num_mols_z])
        box_half_lengths = num_mols * self.molecule_box_length.value / 2
        box_coordinates = np.array([-box_half_lengths, box_half_lengths])
        return box_coordinates

    def _calculate_box_lengths(self) -> np.ndarray:
        return self.box_coordinates[1] - self.box_coordinates[0]

    def _calculate_molecule_box_length(self):
        return self.volume_per_molecule ** (1 / 3)

    def _calculate_volume_per_molecule(self) -> duq.Quantity:
        """
        Calculate the volume per molecule of water, based on the calculated density.

        Returns
        -------
        volume_per_molecule : duq.Quantity
            Volume of a single water molecule, as a `duq.Quantity` object.
        """
        molar_density = self.density / self._mass_molar_h2o
        number_density = molar_density * duq.predefined_constants.avogadro_const
        volume_per_molecule = 1 / number_density
        volume_per_molecule.convert_unit(self.unit_positions ** 3, inplace=True)
        return volume_per_molecule

    def _calculate_density(self) -> duq.Quantity:
        """
        Calculate the density of the system, based on the given temperature and pressure.

        Returns
        -------
        density : duq.Quantity
            Density of the system, as a `duq.Quantity` object.
        """
        # TODO (calculate density as a function of temperature and pressure)
        density = duq.Quantity(997, "kg.m^-3")
        return density

    def _calculate_atomic_numbers(self) -> np.ndarray:
        """
        Create an array containing the atomic number of each atom in the system.

        Returns
        -------
        atomic_numbers : numpy.ndarray
            Atomic numbers of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is
            the number of all atoms.
        """
        return np.tile([8, 1, 1], self._num_molecules_total)

    def _calculate_molecule_ids(self) -> np.ndarray:
        """
        Create a molecule-ID for each atom, specifying to which molecule it belongs.

        Returns
        -------
        molecule_ids : numpy.ndarray
            Molecule-IDs of all atoms in the system, as a 1D-array of shape (n, ), where 'n' is the
            number of atoms in the system. Molecule-IDs start at 0 for the first molecule, and are
            incremented by 1.
        """
        return np.repeat(np.arange(self._num_molecules_total), 3)

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
        connectivity_matrix = np.full((self._num_atoms_total, self._num_atoms_total), False)
        for atom_idx in range(0, self._num_atoms_total, 3):
            connectivity_matrix[atom_idx, atom_idx + 1 : atom_idx + 3] = True
        return connectivity_matrix

    def _generate_single_molecule(self) -> np.ndarray:
        """
        Generate the coordinates for atoms in a molecule of water, where the molecule is in the
        xy-plane, the geometric center is on the origin, the c2-rotation axis is the y-axis, and
        the hydrogen atoms have both the same distance from the oxygen, with negative y-values.

        Returns
        -------
        centered_coordinates : numpy.ndarray
            2D-array of shape (3, 3), where each element is the vector of xyz-coordinates for one
            atom. The first element is the coordinates of oxygen, and the next two elements are of
            hydrogen atoms.
        """
        coordinates = np.zeros((3, 3))
        angle = self._eq_bond_angle.convert_unit("rad").value
        half_angle = angle / 2
        cos = np.cos(half_angle)
        sin = np.sin(half_angle)
        bond_length = self._eq_bond_length.convert_unit(self.unit_positions).value
        x_hydrogens = bond_length * sin
        y_hydrogens = -bond_length * cos
        coordinates[1, :2] = [-x_hydrogens, y_hydrogens]
        coordinates[2, :2] = [x_hydrogens, y_hydrogens]
        geometric_center = (np.max(coordinates, axis=0) + np.min(coordinates, axis=0)) / 2
        centered_coordinates = coordinates - geometric_center
        return centered_coordinates

    def _create_log(self):
        # TODO
        return

    def save_log_to_file(self, filepath):
        # TODO
        return
