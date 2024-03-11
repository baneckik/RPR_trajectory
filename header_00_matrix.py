import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def get_random_schic(size, contacts):
    """
    Creates a random imitataion of a scHi-C
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


def get_main_matrix(sc_matrices, mode="diag"):
    """
    Function creates a matrix from a list of ordered sc_matrices.
    :param sc_matrices: Ordered list or tuple of the 2D numpy arrays of the same size
    :param mode: either "diag" or "random". Specifies how the trans matrices are going to be created.
    :return: 2D numpy array
    """
    n = len(sc_matrices)
    size = sc_matrices[0].shape[0]
    contacts = np.sum(sc_matrices[0])/4
    row_matrices = []
    for i in range(n):
        if mode == "random":
            row_matrices.append(np.concatenate(
                [np.zeros((size, size)) for j in range(i-1)] +
                [get_random_schic(size, contacts) for j in range(int(i != 0))] +
                [sc_matrices[i]] +
                [get_random_schic(size, contacts) for j in range(int(i != n-1))] +
                [np.zeros((size, size)) for j in range(n - i - 2)], axis=1))
        elif mode == "diag":
            row_matrices.append(np.concatenate(
                [np.zeros((size, size)) for j in range(i - 1)] +
                [np.eye(size) for j in range(int(i != 0))] +
                [sc_matrices[i]] +
                [np.eye(size) for j in range(int(i != n - 1))] +
                [np.zeros((size, size)) for j in range(n - i - 2)], axis=1))
        else:
            raise (Exception("Incorrect mode: {}. Must be either diag or random.".format(mode)))
    return np.concatenate(row_matrices, axis=0)


def matrix_plot(matrix, file_name):
    """
    The function visualises the scHi-C matrix and saves the result to the file_name.
    :param matrix: 2D numpy array representing the scHi-C matrix to visualise
    :param file_name: path to the output image
    :return: None
    """
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
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
