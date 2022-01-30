"""
Module containing class TrajectoryAnalyzer used for calculating and plotting properties from trajectory data.
"""

# 3rd-party packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython import display
import duq

# Self
from .data.elements_data import data

mpl.rcParams["figure.dpi"] = 200


class TrajectoryAnalyzer:
    @classmethod
    def from_mdsim_object(cls, mdsim):
        return cls(
            mdsim.positions,
            mdsim.velocities,
            mdsim.timestamps,
            mdsim.atomic_numbers,
            mdsim.molecule_ids,
            mdsim.bonded_atoms_indices,
            mdsim.masses,
            mdsim.energy_potential_coulomb,
            mdsim.energy_potential_lennard_jones,
            mdsim.energy_potential_bond_vibration,
            mdsim.energy_potential_angle_vibration,
            mdsim.distances_interatomic,
            mdsim.bond_angles,
            duq.Unit("Da"),
            mdsim._init_unit_length,
            mdsim._init_unit_time,
        )

    def __init__(
        self,
        positions,
        velocities,
        timestamps,
        atomic_numbers,
        molecule_ids,
        connectivity_matrix,
        masses,
        energy_potential_coulomb,
        energy_potential_lennard_jones,
        energy_potential_bond_vibration,
        energy_potential_angle_vibration,
        distances_interatomic,
        angles_bonds,
        unit_mass,
        unit_length,
        unit_time,
    ):
        """

        Parameters
        ----------
        positions
        velocities
        timestamps
        atomic_numbers : numpy.ndarray
            1D array of length `n` (n = number of atoms) containing the atomic numbers (int) of
            all atoms in the same order as they apper in each frame of position and velocity data.
        molecule_ids : numpy.ndarray
            1D array of length `n` (n = number of atoms) containing the molecule-IDs of all atoms.
            They can have arbitrary values, but should have the same value for all atoms in the same molecule.
        connectivity_matrix : numpy.ndarray
            Connectivity of each atom to other atoms, as a boolean upper-triangular matrix of shape
            (n, n), where 'n' is the total number of atoms.
        masses
        energy_potential_coulomb
        energy_potential_lennard_jones
        energy_potential_bond_vibration
        energy_potential_angle_vibration
        distances_interatomic
        angles_bonds
        unit_mass
        unit_length
        unit_time
        """

        self.positions = positions
        self.velocities = velocities
        self.timestamps = timestamps
        self.atomic_numbers = atomic_numbers
        self.molecule_ids = molecule_ids
        self.connectivity_matrix = connectivity_matrix
        self.masses = masses
        self.energy_potential_coulomb = energy_potential_coulomb
        self.energy_potential_lennard_jones = energy_potential_lennard_jones
        self.energy_potential_bond_vibration = energy_potential_bond_vibration
        self.energy_potential_angle_vibration = energy_potential_angle_vibration
        self.distances_interatomic = distances_interatomic
        self.angles_bonds = angles_bonds

        self._unit_mass = unit_mass
        self._unit_length = unit_length
        self._unit_time = unit_time

        self._unit_velocity = None
        self._unit_momentum = None
        self._unit_energy = None

        self._unit_temperature = duq.Unit("K")

        self._speeds = None
        self._momenta = None
        self._energy_kinetic_per_atom = None
        self._temperature = None
        self._energy_kinetic_total = None
        self._energy_potential_total = None
        self._energy_total = None

    @property
    def unit_mass(self):
        if isinstance(self._unit_mass, str):
            self._unit_mass = duq.Unit(self._unit_mass)
        else:
            pass
        return self._unit_mass

    @property
    def unit_length(self):
        if isinstance(self._unit_length, str):
            self._unit_length = duq.Unit(self._unit_length)
        else:
            pass
        return self._unit_length

    @property
    def unit_time(self):
        if isinstance(self._unit_time, str):
            self._unit_time = duq.Unit(self._unit_time)
        else:
            pass
        return self._unit_time

    @property
    def unit_velocity(self):
        if self._unit_velocity is None:
            self._unit_velocity = self.unit_length / self.unit_time
        else:
            pass
        return self._unit_velocity

    @property
    def unit_momentum(self):
        if self._unit_momentum is None:
            self._unit_momentum = self.unit_mass * self.unit_velocity
        else:
            pass
        return self._unit_momentum

    @property
    def unit_energy(self):
        if self._unit_energy is None:
            self._unit_energy = self.unit_mass * self.unit_length ** 2 / self.unit_time ** 2
        else:
            pass
        return self._unit_energy

    @property
    def unit_temperature(self):
        return self._unit_temperature

    @property
    def energy_total(self):
        if self._energy_total is None:
            self._calculate_energy_total()
        else:
            pass
        return self._energy_total

    @property
    def energy_potential_total(self):
        if self._energy_potential_total is None:
            self._calculate_energy_potential_total()
        else:
            pass
        return self._energy_potential_total

    @property
    def energy_kinetic_total(self):
        if self._energy_kinetic_total is None:
            self._calculate_energy_kinetic_total()
        else:
            pass
        return self._energy_kinetic_total

    @property
    def temperature(self):
        if self._temperature is None:
            self._calculate_temperature()
        else:
            pass
        return self._temperature

    @property
    def energy_kinetic_per_atom(self):
        if self._energy_kinetic_per_atom is None:
            self._calculate_energy_kinetic_per_atom()
        else:
            pass
        return self._energy_kinetic_per_atom

    @property
    def momenta(self):
        if self._momenta is None:
            self._calculate_momenta()
        else:
            pass
        return self._momenta

    @property
    def speeds(self):
        if self._speeds is None:
            self._calculate_speeds()
        else:
            pass
        return self._speeds

    def convert_unit_energy(self, unit):
        # TODO
        pass

    def convert_unit_temperature(self, unit):
        # TODO
        pass

    def convert_unit_velocity(self, unit):
        # TODO
        pass

    def convert_unit_time(self, unit):
        # TODO
        pass

    def _calculate_energy_total(self):
        self._energy_total = self.energy_potential_total + self.energy_kinetic_total
        return

    def _calculate_energy_potential_total(self):
        self._energy_potential_total = (
            self.energy_potential_coulomb
            + self.energy_potential_lennard_jones
            + self.energy_potential_bond_vibration
            + self.energy_potential_angle_vibration
        )
        return

    def _calculate_energy_kinetic_total(self):
        self._energy_kinetic_total = self.energy_kinetic_per_atom.sum(axis=1)
        return

    def _calculate_energy_kinetic_per_atom(self):
        self._energy_kinetic_per_atom = 0.5 * self.momenta * self.speeds
        return

    def _calculate_momenta(self):
        self._momenta = self.masses * self.speeds
        return

    def _calculate_speeds(self):
        self._speeds = np.linalg.norm(self.velocities, axis=2)
        return

    def _calculate_temperature(self):
        unit_boltzmann_const = self.unit_energy / duq.Unit("K")
        boltzmann_const = duq.predefined_constants.boltzmann_const.convert_unit(
            unit_boltzmann_const
        )
        self._temperature = self.energy_kinetic_per_atom.mean(axis=1) / (
            (3 / 2) * boltzmann_const.value
        )
        return

    def plot_speeds(self, figure_dpi=200):
        self.change_plots_dpi(figure_dpi)
        plt.plot(self.timestamps, self.speeds, lw=0.4)
        plt.ylabel(f"Spped [{self.unit_velocity.symbol_as_is}]")
        plt.xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        plt.show()
        return

    def plot_temperature(self, figure_dpi=200):
        self.change_plots_dpi(figure_dpi)
        plt.plot(self.timestamps, self.temperature, lw=0.5)
        plt.ylabel(f"Temp. [{self.unit_temperature.symbol_as_is}]")
        plt.xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        plt.show()
        return

    def plot_energy(self, keywords=("tot", "kin", "pot"), scale="symlog", figure_dpi=200):

        keys = {
            "tot": [self.energy_total, "Total"],
            "kin": [self.energy_kinetic_total, "Kinetic"],
            "pot": [self.energy_potential_total, "Potential"],
            "coulomb": [self.energy_potential_coulomb, "Coulomb"],
            "lj": [self.energy_potential_lennard_jones, "L-J"],
            "bond": [self.energy_potential_bond_vibration, "Bond"],
            "angle": [self.energy_potential_angle_vibration, "Angle"],
        }
        self.change_plots_dpi(figure_dpi)

        keywords = (keywords,) if isinstance(keywords, str) else keywords
        for key in tuple(keywords):
            plt.plot(self.timestamps, keys[key][0], label=keys[key][1], lw=0.6)
        plt.ylabel(f"Energy [{self.unit_energy.symbol_as_is}]")
        plt.xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        plt.legend()
        plt.yscale(scale)
        plt.show()
        return

    def plot_angles(self, figure_dpi=200):
        self.change_plots_dpi(figure_dpi)
        plt.plot(self.timestamps, self.angles_bonds, lw=0.4)
        plt.ylabel(f"Angle [rad]")
        plt.xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        plt.show()
        return

    def plot_distances_bonded_atoms(self, figure_dpi=200):
        self._select_distances_bonded_atoms()
        self.change_plots_dpi(figure_dpi)
        plt.plot(self.timestamps, self.distances_bonded_atoms, lw=0.4)
        plt.ylabel(f"Bond distance [{self.unit_length.symbol_as_is}]")
        plt.xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        plt.show()
        return

    def plot_trajectory_single_frame_2d(
        self,
        frame,
        axis_bounds=None,
        margin_percent=5,
        text=None,
    ):
        """
        Plot a 2D image (only x and y coordinates) of a single frame
        of the trajectory, showing all atoms and bonds.

        Parameters
        ----------
        frame : numpy.ndarray
            2D array of shape (n, m), where `n` is the number of atoms
            and `m` is the number of coordinates (should be at least 2;
            higher dimensions will be ignored). The array contains all
            the coordinates of all atoms at a single time.
        axis_bounds : tuple (optional)
            Tuple of four numbers, defining the lower and upper bounds of
            x- and y-axes, respectively.
        margin_percent : float (optional; default: 5)
            When `axis_bound` is not provided, `margin_percent` is used to
            add an empty margin to each of the four sides of the plot.
            The margin size in each direction is defined as the difference
            between the highest value and lowest value in that direction,
            multiplied with `margin_percent` in percent.
        text : str
            Text to be shown at the top left of the graph.

        Returns
        -------
            None
        """

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")

        if axis_bounds is not None:
            x_min, x_max, y_min, y_max = axis_bounds
            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
        else:
            x_min = np.amin(frame[:, 0])
            x_max = np.amax(frame[:, 0])
            x_margin = (x_max - x_min) * margin_percent / 100
            y_min = np.amin(frame[:, 1])
            y_max = np.amax(frame[:, 1])
            y_margin = (y_max - y_min) * margin_percent / 100
            ax.set(
                xlim=(x_min - x_margin, x_max + x_margin),
                ylim=(y_min - y_margin, y_max + y_margin),
            )

        plt.text(
            0.02,
            0.98,
            text,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        max_len = max(x_max - x_min, y_max - y_min)

        # plot bonds as lines
        for idx_atom, connectivity_mask in enumerate(self.connectivity_matrix):
            pos_curr_atom = frame[idx_atom]
            for pos_connected_atom in frame[connectivity_mask]:
                ax.plot(
                    [pos_curr_atom[0], pos_connected_atom[0]],
                    [pos_curr_atom[1], pos_connected_atom[1]],
                    lw=150 / max_len,
                    color="black",
                )

        # plot atoms as circles
        for mol in np.unique(self.molecule_ids):
            mask = np.where(self.molecule_ids == mol)
            atom_coords = frame[mask]
            atom_types = self.atomic_numbers[mask]
            for idx, atom in enumerate(atom_coords):
                circle = plt.Circle(
                    (atom[0], atom[1]),
                    float(data[atom_types[idx]]["cov_radius"].split()[0]),
                    color=data[atom_types[idx]]["color"],
                    zorder=100 - atom_types[idx],
                )
                ax.add_patch(circle)

        plt.show()
        return

    def plot_trajectory_animation_2d(
        self,
        from_step=0,
        to_step=-1,
        every_n_steps=1,
        margin_percent=0.05,
        fix_frame_limits=False,
        figure_dpi=100,
    ):
        self.change_plots_dpi(figure_dpi)
        if fix_frame_limits:
            x_min = np.amin(self.positions[:, :, 0])
            x_max = np.amax(self.positions[:, :, 0])
            y_min = np.amin(self.positions[:, :, 1])
            y_max = np.amax(self.positions[:, :, 1])
            x_margin = 5  # (x_max - x_min) * margin_percent / 100
            y_margin = 5  # (y_max - y_min) * margin_percent / 100
            axis_bounds = (
                x_min - x_margin,
                x_max + x_margin,
                y_min - y_margin,
                y_max + y_margin,
            )
        else:
            axis_bounds = None

        for idx, frame in enumerate(self.positions[from_step:to_step:every_n_steps]):
            self.plot_trajectory_single_frame_2d(
                frame,
                axis_bounds=axis_bounds,
                text=str(idx * every_n_steps),
            )
            display.clear_output(wait=True)
        return

    def _select_distances_bonded_atoms(self):
        atom_idx = 0
        lis = []
        for bonded_atoms_indices in self.connectivity_matrix[::3]:
            for bonded_atom_idx in bonded_atoms_indices:
                lis.append((atom_idx, bonded_atom_idx))
            atom_idx += 3
        self.distances_bonded_atoms = np.zeros((self.distances_interatomic.shape[0], len(lis)))
        for ind, x in enumerate(self.distances_interatomic):
            for ind2, idx in enumerate(lis):
                self.distances_bonded_atoms[ind, ind2] = x[idx[0], idx[1]]
        return

    @staticmethod
    def change_plots_dpi(dpi):
        mpl.rcParams["figure.dpi"] = dpi
        return
