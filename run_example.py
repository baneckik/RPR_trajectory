from header_00_matrix import *
from header_01_graph import *
from header_02_diagnostics import plot_shepard
import time
from points_io import save_points_as_pdb
import os


def run_simulation(matrices_path, output_folder, random=False, n=4, size=50):
    """
    Runs the RPR trajectory simulation.
    :param matrices_path: The folder of the matrices (if random=False)
    :param output_folder: The folder to store the simulation results.
    :param random: If True random scHi-C matrices are generates. If False matrices from matrices_path are used.
    :param n: Number of scHi-C matrices. Ignored if random=False.
    :param size: Size of the scHi-C matrices.
    :return: main_matrix - numpy array combined scHi-C matrix,
            main_graph - main graph in networkx format,
            distance_matrix - numpy array of distance matrix,
            models - numpy array of the models combined
    """
    start_time = time.time()

    if random:
        matrices = [get_random_schic(size, size ** 2) for i in range(n)]
    else:
        files = os.listdir(matrices_path)
        files = [file for file in files if file.endswith("_chr1.npy")]
        files.sort()
        matrices = [np.load(os.path.join(matrices_path, file)) for file in files]
    size = matrices[0].shape[0]
    n = len(matrices)

    matrix = get_main_matrix(matrices, "interp")
    matrix_plot(matrix, os.path.join(output_folder, "main_matrix.png"), grid_frames=n, highlight_diag=True)
    end_time = time.time()
    print("Elapsed time of creating an initial matrix:", round(end_time - start_time, 2), "seconds")

    start_time = time.time()
    graph = get_graph_from_main_matrix(matrix)
    plot_graph(graph, os.path.join(output_folder, "main_graph.png"))
    end_time = time.time()
    print("Elapsed time of creating a graph:", round(end_time - start_time, 2), "seconds")

    start_time = time.time()
    paths = get_shortest_paths(graph)
    matrix_plot(paths, os.path.join(output_folder, "distance_matrix.png"), grid_frames=n)
    end_time = time.time()
    print("Elapsed time of finding shortest paths:", round(end_time - start_time, 2), "seconds")

    start_time = time.time()
    model = get_mds_model(paths)
    models = [model[size * i:size * (i + 1), :] for i in range(n)]
    end_time = time.time()
    print("Elapsed time of fitting MDS model:", round(end_time - start_time, 2), "seconds")

    frames_path = os.path.join(output_folder, "frames")
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    for i in range(n):
        save_points_as_pdb(models[i],
                           os.path.join(frames_path, "frame_{}.pdb".format(str(i).zfill(2))))

    plot_shepard(paths, models, os.path.join(output_folder, "plot_shepard.png"))

    return matrix, graph, paths, models


if __name__ == "__main__":

    # main_matrix, main_graph, distance_matrix, frame_models = run_simulation(
    #     "examples", "results/test01",
    #     random=True, n=5, size=50)

    main_matrix, main_graph, distance_matrix, frame_models = run_simulation(
        "examples", "results/test02",
        random=False, n=3, size=50)
