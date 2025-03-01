import numpy as np

def PRotUp(B, k, s1):
    """
    Calculates the partial rotation of matrix B by k steps with diagonals Dk and size s1.

    Parameters:
    B (numpy array): The input matrix to be rotated.
    k (int): The number of steps to rotate the matrix.
    s1 (int): The size of the matrix.

    Returns:
    numpy array: The partially rotated matrix.
    """

    # Define the diagonal matrix Dk
    Dk = np.zeros((s1, s1))
    for j in range(s1):
        for i in range(s1):
            if 0 <= i < s1 - k:
                Dk[i, j] = 1

    # Calculate the matrix product of B and Dk
    B_Dk = np.multiply(B, Dk)

    # Calculate the matrix B - B_Dk
    B_minus_B_Dk = np.subtract(B, B_Dk)

    # Calculate the matrix product of RotUp(B_minus_B_Dk, 1)
    RotUp_B_minus_B_Dk = np.rot90(B_minus_B_Dk, 1, axes=(1, 0))

    # Calculate the partially rotated matrix
    PRotUp = np.add(B_Dk, RotUp_B_minus_B_Dk)

    return PRotUp