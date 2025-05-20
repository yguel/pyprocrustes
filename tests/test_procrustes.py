import unittest
from pyprocrustes import compute_transformation, to_homogeneous_points, rotationMatrix3x3
import numpy as np

def verify_transformation(T, R1_points, R2_points):
    """
    Verify that the transformation matrix T transforms R1_points to R2_points.
    
    Parameters
    ----------
    T : np.ndarray
        The transformation matrix.
    R1_points : np.ndarray
        The first set of points.
    R2_points : np.ndarray
        The second set of points.
    
    Returns
    -------
    bool
        True if the transformation is valid, False otherwise.
    """
    # Add 4th coordinate to R1_points[i]
    homogeneous_R1_points = to_homogeneous_points(R1_points)
    
    for i in range(len(R1_points)):
        TR1_pt = T @ homogeneous_R1_points[i]
        if not np.allclose(TR1_pt[:3], R2_points[i], atol=1e-5):
            return False
    return True


class TestComputeTransformation(unittest.TestCase):
    def test_valid_input(self):
        R1_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        R2_points = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1]])
        T, unique = compute_transformation(R1_points, R2_points)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(unique)

    def test_invalid_shape(self):
        R1_points = np.array([[0, 0], [1, 0]])
        R2_points = np.array([[1, 1], [2, 1]])
        with self.assertRaises(ValueError):
            compute_transformation(R1_points, R2_points)

    def test_different_number_of_points(self):
        R1_points = np.array([[0, 0, 0], [1, 0, 0]])
        R2_points = np.array([[1, 1, 1]])
        with self.assertRaises(ValueError):
            compute_transformation(R1_points, R2_points)
            
    def test_90_degree_rotation_aroundZ(self):
        R1_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        R2_points = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
        T, unique = compute_transformation(R1_points, R2_points)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(unique)
        correct_t = np.array([[0, 1, 0, 0],
                     [-1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T[:3, :3], correct_t[:3, :3], decimal=5)
        #Verify that T*R1_points[i] = R2_points[i]
        self.assertTrue(verify_transformation(T, R1_points, R2_points))
        
    def test_180_degree_rotation_aroundZ(self):
        R1_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        R2_points = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0]])
        T, unique = compute_transformation(R1_points, R2_points)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(unique)
        correct_t = np.array([[-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(T[:3, :3], correct_t[:3, :3], decimal=5)
        self.assertTrue(verify_transformation(T, R1_points, R2_points))
        
    def test_45_degree_rotation_aroundZ_plus_translation_1_2_3(self):
        R1_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        R2_points = np.array([[1, 2, 3], [1.70710678, 2.70710678, 3], [0.29289322, 2.70710678, 3]])
        T, unique = compute_transformation(R1_points, R2_points)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(unique)
        correct_t = np.array([[0.70710678, -0.70710678, 0., 1.],
                     [0.70710678, 0.70710678, 0., 2.],
                     [0., 0., 1., 3.],
                     [0., 0., 0., 1.]])
        np.testing.assert_array_almost_equal(T[:3, :4], correct_t[:3, :4], decimal=5)
        self.assertTrue(verify_transformation(T, R1_points, R2_points))
        
    def test_random_rotation_plus_random_translation(self):
        R1_points = np.random.rand(100, 3)
        # Random rotation matrix
        theta = np.random.rand() * 2 * np.pi
        axis = np.random.rand(3)
        
        rotation_matrix = rotationMatrix3x3(theta, axis) 
        
        # Random translation vector
        translation_vector = np.random.rand(3)
        R2_points = (R1_points @ rotation_matrix.T) + translation_vector
        T, unique = compute_transformation(R1_points, R2_points)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(unique)
        self.assertTrue(verify_transformation(T, R1_points, R2_points))


if __name__ == '__main__':
    unittest.main()