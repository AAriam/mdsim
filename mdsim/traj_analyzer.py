"""
Module containing class TrajectoryAnalyzer used for calculating and plotting properties from
trajectory data.
"""

# Standard library
from typing import Union
from pathlib import Path

# 3rd-party packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython import display
import duq

# Self
from .data.elements_data import data
from . import helpers


mpl.rcParams["figure.dpi"] = 200


class TrajectoryAnalyzer:
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        atomic_numbers: np.ndarray,
        molecule_ids: np.ndarray,
        connectivity_matrix: np.ndarray,
        energy_potential_coulomb: np.ndarray,
        energy_potential_lennard_jones: np.ndarray,
        energy_potential_bond_vibration: np.ndarray,
        energy_potential_angle_vibration: np.ndarray,
        distances_interatomic: np.ndarray,
        bond_angles: np.ndarray,
        unit_length: Union[str, duq.Unit],
        unit_time: Union[str, duq.Unit],
        unit_velocity: Union[str, duq.Unit],
        unit_energy: Union[str, duq.Unit],
        unit_angle: Union[str, duq.Unit],
        pbc_box_lengths: np.ndarray = None,
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
            They can have arbitrary values, but should have the same value for all atoms in the
            same molecule.
        connectivity_matrix : numpy.ndarray
            Connectivity of each atom to other atoms, as a boolean upper-triangular matrix of shape
            (n, n), where 'n' is the total number of atoms.
        energy_potential_coulomb
        energy_potential_lennard_jones
        energy_potential_bond_vibration
        energy_potential_angle_vibration
        distances_interatomic
        bond_angles
        unit_length : Union[str, duq.Unit]
            Length-unit of the data.
        unit_time : Union[str, duq.Unit]
            Time-unit of the data.
        """

        self._positions = positions
        self._velocities = velocities
        self._timestamps = timestamps
        self._atomic_numbers = atomic_numbers
        self._molecule_ids = molecule_ids
        self._connectivity_matrix = connectivity_matrix
        self._energy_potential_coulomb = energy_potential_coulomb
        self._energy_potential_lennard_jones = energy_potential_lennard_jones
        self._energy_potential_bond_vibration = energy_potential_bond_vibration
        self._energy_potential_angle_vibration = energy_potential_angle_vibration
        self._distances_interatomic = distances_interatomic
        self._bond_angles = bond_angles

        self._masses = np.array(
            list(
                map(
                    lambda z: data[z]["mass_Da"],
                    self.atomic_numbers,
                )
            )
        )
        self._unit_mass = duq.Unit("Da")

        self._unit_length = helpers.convert_to_unit(unit_length, "L", "unit_length")
        self._unit_time = helpers.convert_to_unit(unit_time, "T", "unit_time")
        self._unit_velocity = helpers.convert_to_unit(unit_velocity, "L.T^-1", "unit_velocity")
        self._unit_energy = helpers.convert_to_unit(unit_energy, "E", "unit_energy")
        self._unit_angle = helpers.convert_to_unit(unit_angle, "dimensionless", "unit_angle")

        self._pbc_box_lengths = pbc_box_lengths

        self._unit_momentum = self.unit_mass * self.unit_velocity
        self._unit_temperature = duq.Unit("K")

        self._speeds = None
        self._momenta = None
        self._energy_kinetic_per_atom = None
        self._temperature = None
        self._energy_kinetic_total = None
        self._energy_potential_total = None
        self._energy_total = None
        self._bond_lengths = None
        return

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def atomic_numbers(self):
        return self._atomic_numbers

    @property
    def molecule_ids(self):
        return self._molecule_ids

    @property
    def connectivity_matrix(self):
        return self._connectivity_matrix

    @property
    def masses(self):
        return self._masses

    @property
    def energy_potential_coulomb(self):
        return self._energy_potential_coulomb

    @property
    def energy_potential_lennard_jones(self):
        return self._energy_potential_lennard_jones

    @property
    def energy_potential_bond_vibration(self):
        return self._energy_potential_bond_vibration

    @property
    def energy_potential_angle_vibration(self):
        return self._energy_potential_angle_vibration

    @property
    def distances_interatomic(self):
        return self._distances_interatomic

    @property
    def bond_angles(self):
        return self._bond_angles

    @property
    def unit_mass(self):
        return self._unit_mass

    @unit_mass.setter
    def unit_mass(self, new_unit_mass):
        conv_shift, conv_factor, self._unit_mass = self.unit_mass.convert_to(new_unit_mass)
        self._masses *= conv_factor
        return

    @property
    def unit_length(self):
        return self._unit_length

    @unit_length.setter
    def unit_length(self, new_unit_length):
        conv_shift, conv_factor, self._unit_length = self._unit_length.convert_to(new_unit_length)
        self._positions *= conv_factor
        return

    @property
    def unit_time(self):
        return self._unit_time

    @unit_time.setter
    def unit_time(self, new_unit_time):
        conv_shift, conv_factor, self._unit_time = self._unit_time.convert_to(new_unit_time)
        self._timestamps *= conv_factor
        return

    @property
    def unit_velocity(self):
        return self._unit_velocity

    @unit_velocity.setter
    def unit_velocity(self, new_unit_velocity):
        conv_shift, conv_factor, self._unit_velocity = self._unit_velocity.convert_to(
            new_unit_velocity
        )
        self._velocities *= conv_factor
        if self._speeds is not None:
            self._speeds *= conv_factor
        return

    @property
    def unit_momentum(self):
        return self._unit_momentum

    @unit_momentum.setter
    def unit_momentum(self, new_unit_momentum):
        conv_shift, conv_factor, self._unit_momentum = self._unit_momentum.convert_to(
            new_unit_momentum
        )
        self._momenta *= conv_factor
        return

    @property
    def unit_energy(self):
        return self._unit_energy

    @unit_energy.setter
    def unit_energy(self, new_unit_energy):
        conv_shift, conv_factor, self._unit_energy = self._unit_energy.convert_to(new_unit_energy)
        self._energy_potential_coulomb *= conv_factor
        self._energy_potential_lennard_jones *= conv_factor
        self._energy_potential_bond_vibration *= conv_factor
        self._energy_potential_angle_vibration *= conv_factor
        if self._energy_potential_total is not None:
            self._energy_potential_total *= conv_factor
        if self._energy_kinetic_total is not None:
            self._energy_kinetic_total *= conv_factor
        if self._energy_total is not None:
            self._energy_total *= conv_factor
        if self._energy_kinetic_per_atom is not None:
            self._energy_kinetic_per_atom *= conv_factor
        return

    @property
    def unit_temperature(self):
        return self._unit_temperature

    @unit_temperature.setter
    def unit_temperature(self, new_unit_temperature):
        conv_shift, conv_factor, self._unit_temperature = self._unit_temperature.convert_to(
            new_unit_temperature
        )
        self._temperature += conv_shift
        return

    @property
    def energy_total(self):
        if self._energy_total is None:
            self._calculate_energy_total()
        return self._energy_total

    @property
    def energy_potential_total(self):
        if self._energy_potential_total is None:
            self._calculate_energy_potential_total()
        return self._energy_potential_total

    @property
    def energy_kinetic_total(self):
        if self._energy_kinetic_total is None:
            self._calculate_energy_kinetic_total()
        return self._energy_kinetic_total

    @property
    def temperature(self):
        if self._temperature is None:
            self._calculate_temperature()
        return self._temperature

    @property
    def energy_kinetic_per_atom(self):
        if self._energy_kinetic_per_atom is None:
            self._calculate_energy_kinetic_per_atom()
        return self._energy_kinetic_per_atom

    @property
    def momenta(self):
        if self._momenta is None:
            self._calculate_momenta()
        return self._momenta

    @property
    def speeds(self):
        if self._speeds is None:
            self._calculate_speeds()
        return self._speeds

    @property
    def bond_lengths(self):
        if self._bond_lengths is None:
            self._bond_lengths = self.distances_interatomic[self.connectivity_matrix]
        return self._bond_lengths

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
        unit_boltzmann_const = self.unit_energy / self.unit_temperature
        boltzmann_const = duq.predefined_constants.boltzmann_const.convert_unit(
            unit_boltzmann_const
        )
        self._temperature = self.energy_kinetic_per_atom.mean(axis=1) / (
            (3 / 2) * boltzmann_const.value
        )
        return

    def _calculate_positions_in_main_box(self):
        if self._pbc_box_lengths is None:
            raise ValueError("PBC box lengths are not given.")
        else:
            pos_in_box = (
                    self.positions - self._pbc_box_lengths *
                    np.rint(self.positions / self._pbc_box_lengths)
            )

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
        self._extract_bond_lengths()
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

    @staticmethod
    def change_plots_dpi(dpi):
        mpl.rcParams["figure.dpi"] = dpi
        return
