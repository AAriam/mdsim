"""
Module containing class TrajectoryAnalyzer used for calculating and plotting properties from trajectory data.
"""

# 3rd-party packages
from async_timeout import enum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import duq
from itertools import cycle
import matplotlib.animation as animation
import IPython.display as display

# from pygame import init

# Self
from .data.elements_data import data
from .helpers import in_jupyter_notebook

mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["animation.embed_limit"] = 2 ** 128


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
        atomic_nums,
        mol_ids,
        bonded_atoms_idx,
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
        atomic_nums : numpy.ndarray
            1D array of length `n` (n = number of atoms) containing the atomic numbers (int) of
            all atoms in the same order as they apper in each frame of position and velocity data.
        mol_ids : numpy.ndarray
            1D array of length `n` (n = number of atoms) containing the molecule-IDs of all atoms.
            They can have arbitrary values, but should have the same value for all atoms in the same molecule.
        bonded_atoms_idx : list
            2D list of length `n` (n = number of atoms) containing the indices (int) of all bonded atoms to each atom.
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
        self.atomic_numbers = atomic_nums
        self.molecule_ids = mol_ids
        self.bonded_atoms_idx = bonded_atoms_idx
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

        self._animation_2d_object = None
        self._ax_plot_objects = None
        self._current_frame_number = None
        self._ax_plot_objects_iter = None
        self._figure = None
        self._current_ax = None
        self._axs = None
        self._is_initial_frame = None
        self._is_animation = None

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

    def _calculate_speeds(self):
        self._speeds = np.linalg.norm(self.velocities, axis=2)
        return

    def _calculate_momenta(self):
        self._momenta = self.masses * self.speeds
        return

    def _calculate_energy_kinetic_per_atom(self):
        self._energy_kinetic_per_atom = 0.5 * self.momenta * self.speeds
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

    def _calculate_energy_kinetic_total(self):
        self._energy_kinetic_total = self.energy_kinetic_per_atom.sum(axis=1)
        return

    def _calculate_energy_potential_total(self):
        self._energy_potential_total = (
            self.energy_potential_coulomb
            + self.energy_potential_lennard_jones
            + self.energy_potential_bond_vibration
            + self.energy_potential_angle_vibration
        )
        return

    def _calculate_energy_total(self):
        self._energy_total = self.energy_kinetic_total + self.energy_potential_total
        return

    def _plot_energy(self, keywords=("tot", "kin", "pot"), scale="symlog"):

        keys = {
            "tot": [self.energy_total, "Total"],
            "kin": [self.energy_kinetic_total, "Kinetic"],
            "pot": [self.energy_potential_total, "Potential"],
            "coulomb": [self.energy_potential_coulomb, "Coulomb"],
            "lj": [self.energy_potential_lennard_jones, "L-J"],
            "bond": [self.energy_potential_bond_vibration, "Bond"],
            "angle": [self.energy_potential_angle_vibration, "Angle"],
        }

        keywords = (keywords,) if isinstance(keywords, str) else keywords
        if self._is_initial_frame:
            for key in tuple(keywords):
                self._current_ax.plot(self.timestamps, keys[key][0], label=keys[key][1], lw=0.6)
            if self._is_animation:
                for key in tuple(keywords):
                    self._ax_plot_objects.append(
                        self._current_ax.plot(self.timestamps[0], keys[key][0][0], "ro")[0]
                    )
            self._current_ax.set_xlabel(f"Time [{self.unit_time.symbol_as_is}]")
            self._current_ax.set_ylabel(f"Energy [{self.unit_energy.symbol_as_is}]")
            self._current_ax.set_yscale(scale)
            plt.legend()
        else:
            for key in tuple(keywords):
                self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_data(
                    self.timestamps[self._current_frame_number],
                    keys[key][0][self._current_frame_number],
                )

        return

    def _plot_speeds(self):
        if self._is_initial_frame:
            self._current_ax.plot(self.timestamps, self.speeds, lw=0.4)
            if self._is_animation:
                for speed in self.speeds.T:
                    self._ax_plot_objects.append(
                        self._current_ax.plot(self.timestamps[0], speed[0], "ro")[0]
                    )
            self._current_ax.set_ylabel(f"Speed [{self.unit_velocity.symbol_as_is}]")
            self._current_ax.set_xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        else:
            for speed in self.speeds.T:
                self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_data(
                    self.timestamps[self._current_frame_number], speed[self._current_frame_number]
                )
        return

    def _plot_distances_bonded_atoms(self):
        if self._is_initial_frame:
            self._select_distances_bonded_atoms()
            self._current_ax.plot(self.timestamps, self.distances_bonded_atoms, lw=0.4)
            if self._is_animation:
                for dba in self.distances_bonded_atoms.T:
                    self._ax_plot_objects.append(
                        self._current_ax.plot(self.timestamps[0], dba[0], "ro")[0]
                    )
            self._current_ax.set_ylabel(f"Bond distance [{self.unit_length.symbol_as_is}]")
            self._current_ax.set_xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        else:
            for dba in self.distances_bonded_atoms.T:
                self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_data(
                    self.timestamps[self._current_frame_number], dba[self._current_frame_number]
                )
        return

    def _plot_temperature(self):
        if self._is_initial_frame:
            self._current_ax.plot(self.timestamps, self.temperature, lw=0.5)
            if self._is_animation:
                self._ax_plot_objects.append(
                    self._current_ax.plot(self.timestamps[0], self.temperature[0], "ro")[0]
                )
            self._current_ax.set_ylabel(f"Temp. [{self.unit_temperature.symbol_as_is}]")
            self._current_ax.set_xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        else:
            self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_data(
                self.timestamps[self._current_frame_number],
                self.temperature[self._current_frame_number],
            )
        return

    def _plot_angles(self):
        if self._is_initial_frame:
            self._current_ax.plot(self.timestamps, self.angles_bonds, lw=0.4)
            if self._is_animation:
                for angle_bond in self.angles_bonds.T:
                    self._ax_plot_objects.append(
                        self._current_ax.plot(self.timestamps[0], angle_bond[0], "ro")[0]
                    )
            self._current_ax.set_ylabel(f"Angle [rad]")
            self._current_ax.set_xlabel(f"Time [{self.unit_time.symbol_as_is}]")
        else:
            for angle_bond in self.angles_bonds.T:
                self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_data(
                    self.timestamps[self._current_frame_number],
                    angle_bond[self._current_frame_number],
                )
        return

    def _plot_simulation(self, frame, axis_bounds=None, margin_percent=5):
        """
        Plot a 2D image (only x and y coordinates) of the initial frame
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
        if self._is_initial_frame:

            self._current_ax.set_aspect("equal")
            x_min = np.amin(frame[:, 0])
            x_max = np.amax(frame[:, 0])
            y_min = np.amin(frame[:, 1])
            y_max = np.amax(frame[:, 1])

            if axis_bounds is not None:
                x_min, x_max, y_min, y_max = axis_bounds
                self._current_ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
            else:
                x_margin = (x_max - x_min) * margin_percent / 100
                y_margin = (y_max - y_min) * margin_percent / 100
                self._current_ax.set(
                    xlim=(x_min - x_margin, x_max + x_margin),
                    ylim=(y_min - y_margin, y_max + y_margin),
                )

            ax_plt_object = self._current_ax.text(
                0.02,
                0.97,
                0,
                horizontalalignment="left",
                verticalalignment="center",
                transform=self._current_ax.transAxes,
            )
            self._ax_plot_objects.append(ax_plt_object)

            max_len = max(x_max - x_min, y_max - y_min)
            # plot bonds as lines
            for idx_atom, idx_bonded_atoms in enumerate(self.bonded_atoms_idx):
                idx_bonded_atoms = np.array(idx_bonded_atoms)
                mask = idx_bonded_atoms > idx_atom
                for idx_bonded_atom in idx_bonded_atoms[mask]:
                    ax_plt_object = self._current_ax.plot(
                        [frame[idx_atom][0], frame[idx_bonded_atom][0]],
                        [frame[idx_atom][1], frame[idx_bonded_atom][1]],
                        lw=150 / max_len,
                        color="black",
                    )
                    self._ax_plot_objects.append(ax_plt_object[0])

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
                    ax_plt_object = self._current_ax.add_patch(circle)
                    self._ax_plot_objects.append(ax_plt_object)
        else:
            self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_text(
                str(self._current_frame_number)
            )
            # plot bonds as lines
            for idx_atom, idx_bonded_atoms in enumerate(self.bonded_atoms_idx):
                idx_bonded_atoms = np.array(idx_bonded_atoms)
                mask = idx_bonded_atoms > idx_atom
                for idx_bonded_atom in idx_bonded_atoms[mask]:
                    self._ax_plot_objects[next(self._ax_plot_objects_iter)].set_data(
                        np.asarray([frame[idx_atom][0], frame[idx_bonded_atom][0]]),
                        np.asarray([frame[idx_atom][1], frame[idx_bonded_atom][1]]),
                    )
            # plot atoms as circles
            for mol in np.unique(self.molecule_ids):
                mask = np.where(self.molecule_ids == mol)
                atom_coords = frame[mask]
                for idx, atom in enumerate(atom_coords):
                    self._ax_plot_objects[next(self._ax_plot_objects_iter)].center = (
                        atom[0],
                        atom[1],
                    )

    def _plot_frame(self, frame, axis_bounds=None, margin_percent=5, plots=("main",), **kwargs):
        for plot_name in plots:
            self._current_ax = self._axs[plot_name]
            if "energy" == plot_name:
                keywords = kwargs.get("energy", ("tot", "kin", "pot"))
                self._plot_energy(keywords=keywords)
            elif "temperature" == plot_name:
                self._plot_temperature()
            elif "angles" == plot_name:
                self._plot_angles()
            elif "speeds" == plot_name:
                self._plot_speeds()
            elif "bonded atoms distances" == plot_name:
                self._plot_distances_bonded_atoms()
            elif "main" == plot_name:
                self._plot_simulation(
                    frame, axis_bounds=axis_bounds, margin_percent=margin_percent
                )

        if (
            self._is_initial_frame and in_jupyter_notebook()
        ):  # This is here to prevent showing this plot after the animation
            plt.show()

        return

    def plot(
        self,
        from_step=0,
        to_step=-1,
        every_n_steps=1,
        interval_between_frames_milliseconds=100,
        animation_duration_seconds=None,
        margin_percent=0.05,
        fix_frame_limits=False,
        figure_dpi=200,
        horizontal_plot=False,
        plots=(
            "energy",
            "angles",
        ),
        **kwargs,
    ):

        self._current_frame_number = from_step
        self._ax_plot_objects = []
        if "main" in plots:
            self._is_animation = True
        else:
            self._is_animation = False

        self.change_plots_dpi(figure_dpi)
        if fix_frame_limits:
            x_min = np.amin(self.positions[:, :, 0])
            x_max = np.amax(self.positions[:, :, 0])
            y_min = np.amin(self.positions[:, :, 1])
            y_max = np.amax(self.positions[:, :, 1])
            x_margin = 5
            y_margin = 5
            axis_bounds = (
                x_min - x_margin,
                x_max + x_margin,
                y_min - y_margin,
                y_max + y_margin,
            )
        else:
            axis_bounds = None

        self._figure = plt.figure(figsize=(20, 10))
        self._axs = {}

        # show subplot vertically or horizontally
        if (("main" in plots) and (len(plots) < 4)) and not horizontal_plot:
            encountered_main = False
            subplot_size = 1 + len(plots)
            for i, plot_name in enumerate(plots):
                if plot_name == "main":
                    ax = self._figure.add_subplot(subplot_size, 1, (i + 1, i + 2))
                    encountered_main = True
                else:
                    if encountered_main:
                        ax = self._figure.add_subplot(subplot_size, 1, i + 2)
                    else:
                        ax = self._figure.add_subplot(subplot_size, 1, i + 1)
                self._axs.update({plot_name: ax})
        else:
            subplot_size = len(plots)
            if "main" in plots:
                subplot_size = len(plots) - 1
            subplot_counter = 1
            for plot_name in plots:
                if plot_name == "main":
                    ax = self._figure.add_subplot(1, 3, (2, 3))
                else:
                    ax = self._figure.add_subplot(subplot_size, 3, subplot_counter)
                    subplot_counter += 3
                self._axs.update({plot_name: ax})

        # init frame
        self._is_initial_frame = True
        self._plot_frame(
            frame=self.positions[from_step],
            axis_bounds=axis_bounds,
            margin_percent=margin_percent,
            plots=plots,
            **kwargs,
        )

        self._is_initial_frame = False

        # animation frame
        def _render(itr, frames, plots):
            frame = frames[itr]
            self._current_frame_number = itr * every_n_steps
            self._ax_plot_objects_iter = cycle(range(len(self._ax_plot_objects)))

            self._plot_frame(
                frame=frame,
                axis_bounds=axis_bounds,
                margin_percent=margin_percent,
                plots=plots,
            )

        if to_step == -1:
            to_step = len(self.positions) - 1
        number_of_frames = len(range(from_step, to_step, every_n_steps))
        if animation_duration_seconds:
            interval_between_frames_milliseconds = (
                animation_duration_seconds * 1000 / number_of_frames
            )

        if self._is_animation:
            self._animation_2d_object = animation.FuncAnimation(
                self._figure,
                _render,
                frames=number_of_frames,
                fargs=(self.positions[from_step:to_step:every_n_steps], plots),
                interval=interval_between_frames_milliseconds,
            )

            if in_jupyter_notebook():
                display.clear_output()
                return display.HTML(self._animation_2d_object.to_jshtml())
            else:
                plt.show()
                return
        else:
            plt.show()

    def save_plot_animation(self, file_address: str = None):

        if file_address:
            self._animation_2d_object.save(filename=file_address)
        else:
            raise ValueError("A file address should be provided.")
        return

    def _select_distances_bonded_atoms(self):
        atom_idx = 0
        lis = []
        for bonded_atoms_indices in self.bonded_atoms_idx[::3]:
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
