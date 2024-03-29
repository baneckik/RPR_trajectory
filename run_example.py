from header_00_matrix import *
from header_01_graph import *
import time
from points_io import save_points_as_pdb


if __name__ == "__main__":
    n = 10
    size = 40

    start_time = time.time()
    matrices = [get_random_schic(size, size**2) for i in range(n)]
    matrix = get_main_matrix(matrices, "interp")
    matrix_plot(matrix, "./test.png", grid_frames=n, highlight_diag=True)
    end_time = time.time()
    print("Elapsed time of creating an initial matrix:", round(end_time - start_time, 2), "seconds")

    start_time = time.time()
    graph = get_graph_from_main_matrix(matrix)
    plot_graph(graph, "./test_graph.png")
    end_time = time.time()
    print("Elapsed time of creating a graph:", round(end_time - start_time, 2), "seconds")

    start_time = time.time()
    paths = get_shortest_paths(graph)
    print(paths)
    matrix_plot(paths, "./test_dist.png", grid_frames=n)
    end_time = time.time()
    print("Elapsed time of finding shortest paths:", round(end_time - start_time, 2), "seconds")

    start_time = time.time()
    model = get_mds_model(paths)
    end_time = time.time()
    print("Elapsed time of fitting MDS model:", round(end_time - start_time, 2), "seconds")

    for i in range(n):
        save_points_as_pdb(model[size*i:size*(i+1), :], "results/test00/test_{}.pdb".format(str(i).zfill(2)))

