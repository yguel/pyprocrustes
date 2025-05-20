import numpy as np


def to_homogeneous_points( pt : np.ndarray ) -> np.ndarray:
    h = np.ones((pt.shape[0], 4))
    # copy pt into h
    for i in range(len(pt)):
        h[i][:3] = pt[i]
    return h

def rotationMatrix3x3( angle: float, axis: np.ndarray ) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for a given angle and axis of rotation.

    Parameters
    ----------
    angle : float
        The angle of rotation in radians.
    axis : np.ndarray
        The axis of rotation, must be a 3D vector.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis / np.linalg.norm(axis)

    return np.array([[t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
                     [t*y*x + s*z, t*y*y + c,   t*y*z - s*x],
                     [t*z*x - s*y, t*z*y + s*x, t*z*z + c]])