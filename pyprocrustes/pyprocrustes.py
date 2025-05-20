import numpy as np
from typing import Tuple

def compute_transformation(R1_points: np.ndarray, R2_points: np.ndarray) -> Tuple[np.ndarray,bool]:
    """
    Compute the homogeneous transformation matrix to align two sets of points.

    Parameters
    ----------
    R1_points : np.ndarray of shape (n, 3)
        The first set of n 3D points, where each row represents a point.
    R2_points : np.ndarray of shape (n, 3)
        The second set of n 3D points, where each row represents a point.
        Must have the same number of points as R1_points.

    Returns
    -------
    Tuple[np.ndarray,bool]
        A tuple containing:
            * The homogeneous matrix T (rotation matrix and translation vector) that transforms a point from R1_points to a point in R2_points. 
            p2 = T @ p1, where p1 is a point from R1_points and p2 is the corresponding point in R2_points.
            * A boolean indicating whether the transformation was unique (True) or not (False), the last case is when points are aligned and many possible solutions exist.

    Raises
    ------
    ValueError
        If the input arrays don't have the correct shape or the same number of points.
    """
    # Validate input dimensions
    if R1_points.ndim != 2 or R1_points.shape[1] != 3:
        raise ValueError(f"R1_points must be a (n, 3) array, got {R1_points.shape}")
    if R2_points.ndim != 2 or R2_points.shape[1] != 3:
        raise ValueError(f"R2_points must be a (n, 3) array, got {R2_points.shape}")
    if R1_points.shape[0] != R2_points.shape[0]:
        raise ValueError(f"Both point sets must have the same number of points, got {R1_points.shape[0]} and {R2_points.shape[0]}")
    
    # Compute the centroids of the two sets of points
    centroid_R1 = np.mean(R1_points, axis=0)
    centroid_R2 = np.mean(R2_points, axis=0)

    # Center the points by subtracting the centroids
    centered_R1 = R1_points - centroid_R1
    centered_R2 = R2_points - centroid_R2

    # Compute the covariance matrix
    covariance_matrix = np.dot(centered_R1.T, centered_R2)

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Compute the expected rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)
    
    unique = True
    # Ensure a proper rotation (det(R) = 1), in case of coplanar or aligned points
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)
        
    # Check if the points are aligned by computing the second min of the absolute values of the singular values
    # Only in the case of aligned points, there will be many solutions
    abs_singular_values = np.abs(S)
    # Sort the singular values, the last one should be the smallest
    abs_singular_values.sort()
    min_2nd_singular_value = abs_singular_values[1]
    if min_2nd_singular_value < 1e-6:
        unique = False

    # Compute the translation vector
    translation_vector = centroid_R2 - np.dot(rotation_matrix, centroid_R1)
    
    # Construct the homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    transformation_matrix[3, 3] = 1.0
    return (transformation_matrix,unique)