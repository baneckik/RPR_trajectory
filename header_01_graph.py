import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


def get_weight(main_matrix, bead_i, bead_j):
    n = main_matrix.shape[0]
    row_i = main_matrix[bead_i, :]
    row_j = main_matrix[bead_j, :]
    intersection = sum([row_i[k] + row_j[k] > 1 for k in range(n)])
    union = sum([row_i[k] + row_j[k] > 0 for k in range(n)])
    return 1 - intersection / union


def get_graph_from_main_matrix(main_matrix):
    n = main_matrix.shape[0]

    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if main_matrix[i, j] != 0:
                w = get_weight(main_matrix, i, j)
                edge_list.append((i, j, w))

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edge_list)

    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    print("Number of edges:", num_edges)
    print("Number of vertices (nodes):", num_nodes)

    return graph


def get_shortest_paths(graph):
    # n = graph.number_of_nodes()
    # lengths_dict = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    # dist_matrix = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         dist_matrix[i, j] = lengths_dict[i][j]
    # return np.array(lengths_dict)
    return nx.floyd_warshall_numpy(graph, weight='weight')


def get_mds_model(distance_matrix):
    embedding = MDS(n_components=3, normalized_stress='auto', dissimilarity='precomputed')
    return embedding.fit_transform(distance_matrix)


def plot_graph(graph, path):
    nx.draw_spring(graph)
    plt.savefig(path)
    plt.close()
