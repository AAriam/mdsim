import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 150


class SodiumChloride:
    # Lattice constant in Å
    _a = 5.6402
    _num_atoms_prim_cell = 8
    # Charges in unit of e
    _charges_prim_cell = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    # Coordinates of the particles in the primitive cell, in unit of a/2
    _primitive_cell = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ]
    )

    def __init__(self, dc=12):
        self._dc = dc
        self._num_prim_cells_1d = int(np.ceil(dc * 2 / self._a))
        self._num_prim_cells_3d = self._num_prim_cells_1d ** 3
        self._num_atoms = self._num_prim_cells_3d * self._num_atoms_prim_cell
        self._positions = self._calculate_positions()

    def _calculate_positions(self) -> np.ndarray:
        # Create an empty array to store the positions
        q = np.zeros((self._num_atoms, 3))
        for i in range(self._num_prim_cells_1d ** 3):
            # Create a translation unit vector, pointing from the origin to the current cell
            trans_unit_vector = np.array(
                [
                    i % self._num_prim_cells_1d,
                    (i // self._num_prim_cells_1d) % self._num_prim_cells_1d,
                    i // (self._num_prim_cells_1d * self._num_prim_cells_1d),
                ]
            )
            # Multiply the unit vector with the box-length of the primitive cell, to get the
            # actual translation vector
            trans_vector = trans_unit_vector * np.array([2, 2, 2])
            # Shift the coordinates of the primitive cell into its current cell and assign it
            q[self._num_atoms_prim_cell * i : self._num_atoms_prim_cell * (i+1)] = (
                    self._primitive_cell + trans_vector
            )
        # Multiply the positions with their unit (i.e. a/2) to bring the positions in Å
        q *= self._a / 2
        # Center the box on the origin
        midpoint = (np.max(q, axis=0) + np.min(q, axis=0)) / 2
        q -= midpoint
        return q

    @property
    def positions(self):
        return self._positions

    @property
    def charges(self):
        return np.tile(self._charges_prim_cell, self._num_prim_cells_3d)

    @property
    def box_lengths(self):
        return np.repeat((self._num_prim_cells_1d * self._a), 3)

    def plot_ensemble(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = np.tile(["r", "r", "r", "r", "b", "b", "b", "b"], self._num_prim_cells_3d)
        ax.scatter(self._positions[:, 0], self._positions[:, 1], self._positions[:, 2], color=colors)
        ax_lim = (-self.box_lengths[0]/2, self.box_lengths[0]/2)
        ax.set_xlim(ax_lim)
        ax.set_ylim(ax_lim)
        ax.set_zlim(ax_lim)
        plt.show()
        return







