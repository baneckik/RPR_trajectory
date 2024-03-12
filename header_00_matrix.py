import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def get_random_schic(size, contacts):
    """
    Creates a random imitation of a scHi-C
    :param size: Integer. Size of a given matrix
    :param contacts: Integer. Approximate number of contacts in the upper part of the matrix
    :return: 2D numpy array
    """
    matrix = np.zeros((size, size))
    pvals_multinomial = np.array([(x + 1) / (size - x) for x in range(size)])
    pvals_multinomial = pvals_multinomial / np.sum(pvals_multinomial)
    diagonals = np.random.multinomial(contacts, pvals_multinomial)
    for i, d in enumerate(diagonals):
        positions = np.random.randint(0, i + 1, d)
        for p in positions:
            matrix[size - 1 - i + p, p] = 1
            matrix[p, size - 1 - i + p] = 1
    return matrix


def get_interp_matrix(matrix1, matrix2):
    """
    Calculates the interpolation scHi-C matrix from two input scHi-C matrices.
    :param matrix1: 2D numpy array
    :param matrix2: 2D numpy array
    :return: 2D numpy array
    """
    n = matrix1.shape[0]
    new_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if matrix1[i, j] > 0 and matrix2[i, j] > 0:
                new_matrix[i, j] = 1
                new_matrix[j, i] = 1
            elif matrix1[i, j] > 0 or matrix2[i, j] > 0:
                new_matrix[i, j] = np.random.randint(2)
                new_matrix[j, i] = new_matrix[i, j]
    return new_matrix


def get_main_matrix(sc_matrices, mode="diag"):
    """
    Function creates a main matrix from a list of ordered sc_matrices.
    :param sc_matrices: Ordered list or tuple of the 2D numpy arrays of the same size
    :param mode: either "interp", "diag" or "random". Specifies how the trans matrices are going to be created.
    :return: 2D numpy array
    """
    n = len(sc_matrices)
    size = sc_matrices[0].shape[0]
    contacts = np.sum(sc_matrices[0])
    row_matrices = []

    if mode == "random":
        for i in range(n):
            row_matrices.append(np.concatenate(
                [np.zeros((size, size)) for j in range(i - 1)] +
                [get_random_schic(size, contacts) for j in range(int(i != 0))] +
                [sc_matrices[i]] +
                [get_random_schic(size, contacts) for j in range(int(i != n - 1))] +
                [np.zeros((size, size)) for j in range(n - i - 2)], axis=1))
    elif mode == "diag":
        for i in range(n):
            row_matrices.append(np.concatenate(
                [np.zeros((size, size)) for j in range(i - 1)] +
                [np.eye(size) for j in range(int(i != 0))] +
                [sc_matrices[i]] +
                [np.eye(size) for j in range(int(i != n - 1))] +
                [np.zeros((size, size)) for j in range(n - i - 2)], axis=1))
    elif mode == "interp":
        row_matrices = [[np.eye(1) for i in range(n)] for j in range(n)]
        for i in range(n):
            row_matrices[i][i] = sc_matrices[i]

        for diag in range(n - 1):
            for j in range(n - diag - 1):
                row_matrices[j][diag + j + 1] = get_interp_matrix(row_matrices[j][diag + j],
                                                                  row_matrices[j + 1][diag + j + 1])
                row_matrices[diag + j + 1][j] = row_matrices[j][diag + j + 1]
        row_matrices = [np.concatenate(row, axis=1) for row in row_matrices]
    else:
        raise (Exception("Incorrect mode: {}. Must be either diag or random.".format(mode)))

    return np.concatenate(row_matrices, axis=0)


def matrix_plot(matrix, file_name, grid_frames=None):
    """
    The function visualises the scHi-C matrix and saves the result to the file_name.
    :param matrix: 2D numpy array representing the scHi-C matrix to visualise
    :param file_name: path to the output image
    :param grid_frames: If not None, integer value indicates the number of sub matrices to pint out on the plot.
    :return: None
    """

    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    if grid_frames is not None:
        n = int(matrix.shape[0]/grid_frames)
        for i in range(1, grid_frames):
            plt.axvline(n * i - 0.5)
            plt.axhline(n * i - 0.5)

    plt.xticks([])
    plt.yticks([])
    plt.savefig(file_name)


def resize_matrix(matrix, new_size):
    """
    Resizes the given matrix to a size (new_size, new_size). The matrix is expected to be square matrix.
    :param matrix: 2D numpy array
    :param new_size: positive integer
    :return: 2D numpy array of a size (new_size, new_size)
    """
    n = matrix.shape[0]
    return zoom(matrix, new_size / n)
