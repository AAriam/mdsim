"""
Helper functions.
"""

# Standard library
from typing import Union

# 3rd-party packages
import numpy as np
import duq


def rotate_3d_vector(vectors, rot_angle_x=0, rot_angle_y=0, rot_angle_z=0, order="xyz"):
    """
    Rotate all 3-dimensional vectors in an array, according to the right-hand-rule.

    Parameters
    ----------
    vectors : numpy.ndarray
        Array of shape (n, 3) containing 'n' 3d-vectors to be rotated.
    rot_angle_x : float (optional; default: 0)
        Rotation angle (in radian) about x-axis.
    rot_angle_y : float (optional; default: 0)
        Rotation angle (in radian) about y-axis.
    rot_angle_z : float (optional; default: 0)
        Rotation angle (in radian) about z-axis.
    order : str (optional; default: "xyz")
        The order of rotation; can be any combination of 'x', 'y' and 'z';
        for example: "yxz", "zx", "x", "xx".
        The rotation is performed from left to right; e.g. "xyz" means first rotate about x-axis,
        then y-axis, and lastly, z-axis.

    Returns
    -------
        numpy.ndarray
        Rotated array of shape (n, 3).
    """
    rot_angles = {"x": rot_angle_x, "y": rot_angle_y, "z": rot_angle_z}
    for axis in order:
        vectors = rotate_3d_vector_around_axis(vectors, rot_angles[axis], axis)
    return vectors


def rotate_3d_vector_around_axis(vectors, rot_angle, axis="z"):
    """
    Rotate all 3-dimensional vectors in an array about a single axis, according to the
    right-hand-rule.

    Parameters
    ----------
    vectors : numpy.ndarray
        Array of shape (n, 3) containing 'n' 3d-vectors to be rotated.
    rot_angle : float
        Rotation angle in radian.
    axis : str (optional; default: "z")
        The axis of rotation; allowed values are 'x', 'y' and 'z'.

    Returns
    -------
        numpy.ndarray
        Rotated array of shape (n, 3).
    """
    sin = np.sin(rot_angle)
    cos = np.cos(rot_angle)
    rot_matrix = {
        "x": np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]]),
        "y": np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]),
        "z": np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]),
    }
    vectors_rotated = np.zeros_like(vectors)
    for i in range(vectors.shape[0]):
        vectors_rotated[i] = np.dot(rot_matrix[axis], vectors[i])
    return vectors_rotated


def convert_to_unit(
    unit: Union[duq.Unit, str], correct_dimension: str, param_name: str
) -> duq.Unit:
    """
    Verify that a given `unit` is either a string or a `duq.Unit` object, and raise an error
    otherwise. If it's a string, transform it to a `duq.Unit` object. Verify that the `duq.Unit`
    object has the expected dimension, and raise an error otherwise.

    Parameters
    ----------
    unit : Union[duq.Unit, str]
        Unit of interest, either as a string representation (see duq.Unit for more details) or
        a `duq.Unit` object.
    correct_dimension : str
        String representation of the expected dimension of the unit (see duq.Dimension for more
        details).
    param_name : str
        Name of the parameter to which `unit` is bound; to be mentioned in the error message.

    Returns
    -------
        duq.Unit
        Object representing the unit of interest.

    Raises
    ------
    ValueError
    """
    if isinstance(unit, str):
        unit_obj = duq.Unit(unit)
    elif isinstance(unit, duq.Unit):
        unit_obj = unit
    else:
        raise ValueError(f"Type of parameter `{param_name}` should be either duq.Unit or string.")
    raise_for_dimension(unit_obj, correct_dimension, param_name)
    return unit_obj


def convert_to_quantity(
    quantity: Union[duq.Quantity, str], correct_dimension: str, param_name: str
) -> duq.Quantity:
    """
    Verify that a given `quantity` is either a string or a `duq.Quantity` object, and raise an
    error otherwise. If it's a string, transform it to a `duq.Quantity` object. Verify that the
    `duq.Quantity` object has the expected dimension, and raise an error otherwise.

    Parameters
    ----------
    quantity : Union[duq.Quantity, str]
        Quantity of interest, either as a string representation (see duq.Quantity for more details)
        or a `duq.Quantity` object.
    correct_dimension : str
        String representation of the expected dimension of the quantity (see duq.Dimension for more
        details).
    param_name : str
        Name of the parameter to which `quantity` is bound; to be mentioned in the error message.

    Returns
    -------
        duq.Quantity
        Object representing the quantity of interest.

    Raises
    ------
    ValueError
    """
    if isinstance(quantity, str):
        quantity_value, quantity_unit = quantity.split()
        quantity_obj = duq.Quantity(float(quantity_value), quantity_unit)
    elif isinstance(quantity, duq.Quantity):
        quantity_obj = quantity
    else:
        raise ValueError(
            f"Type of parameter `{param_name}` should be either duq.Quantity or string."
        )
    raise_for_dimension(quantity_obj, correct_dimension, param_name)
    return quantity_obj


def raise_for_dimension(
    unit_or_quantity: Union[duq.Unit, duq.Quantity],
    correct_dimension: str,
    param_name: str,
) -> None:
    """
    Verify that a `duq.Quantity` or `duq.Unit` object has the correct dimension,
    and raise an error otherwise.

    Parameters
    ----------
    unit_or_quantity : Union[duq.Unit, duq.Quantity]
        Object whose dimension is to be verified.
    correct_dimension : str
        String representation of the expected dimension of the object (see duq.Dimension for more
        details).
    param_name : str
        Name of the parameter to which the `duq.Unit` or `duq.Quantity` object is bound; to be
        mentioned in the error message.

    Returns
    -------
        None

    Raises
    ------
    ValueError
    """
    correct_dim_obj = duq.Dimension(correct_dimension)
    if unit_or_quantity.dimension != correct_dim_obj:
        raise ValueError(
            f"Parameter `{param_name}` should have the physical dimension of "
            f"{correct_dim_obj.name_as_is}."
        )
    return


def raise_num_for_value_and_type(num: any, param_name) -> None:
    """
    Verify that `num` is an integer greater than 1, and raise an error otherwise.

    Parameters
    ----------
    num : any
        Object to be verified.
    param_name : str
        Name of the parameter to which the `num` object is bound; to be mentioned in the error
        message.

    Returns
    -------
        None

    Raises
    ------
    ValueError
    """
    if not isinstance(num, int):
        raise ValueError(f"Parameter `{param_name}` should be an integer.")
    elif num < 1:
        raise ValueError(f"Parameter `{param_name}` should be an integer greater than 1.")
    return


def in_jupyter_notebook() -> bool:
    """
    Check whether the code is running in a Jupyter notebook.

    Returns
    -------
        bool
    """
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        return True  # Jupyter notebook
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    elif shell == "NoneType":
        return False  # Terminal running Python
    else:
        return False  # Other Envs
