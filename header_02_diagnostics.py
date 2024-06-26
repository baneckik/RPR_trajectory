import os

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_shepard(pre_dist_mat, models, output_file):
    """
    Plots a Shepard Diagram comparing distances before and after MDS.
    :param pre_dist_mat:
    :param models:
    :param output_file:
    :return:
    """
    model = np.vstack(models)
    post_dist_mat = distance_matrix(model, model)
    size = models[0].shape[0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    dist_pre, dist_post, cis_trans = [], [], []
    for i in range(model.shape[0]):
        for j in range(i + 1, model.shape[0]):
            dist_pre.append(pre_dist_mat[i, j])
            dist_post.append(post_dist_mat[i, j])
            if i // size == j // size:
                cis_trans.append(0)
            else:
                cis_trans.append(1)
    df = pd.DataFrame({"pre": dist_pre,
                       "post": dist_post,
                       "cis_trans": cis_trans})

    plot0 = ax[0].scatter(dist_pre, dist_post, s=0.1, c=cis_trans, cmap="bwr")
    ax[0].legend(handles=plot0.legend_elements()[0], labels=["cis", "trans"])
    sns.kdeplot(df.sample(200), x="pre", y="post", cmap="Blues", fill=True, thresh=0, levels=10, ax=ax[1])
    sns.kdeplot(df[["post", "pre"]], fill=True, thresh=0, levels=10, ax=ax[2], legend=False)
    ax[0].set_xlabel('Graph distances before MDS')
    ax[0].set_ylabel('Euclidean distances after MDS')
    ax[1].set_xlabel('Graph distances before MDS')
    ax[1].set_ylabel('Euclidean distances after MDS')
    ax[1].set_title('Shepard Diagram')
    ax[2].set_xlabel("Distances")
    plt.legend(title='Distances', loc='upper right', labels=["pre MDS", "post MDS"])

    plt.savefig(output_file)


def plot_trend_matrix(npy_folder):
    file_names = os.listdir(npy_folder)
    file_names.sort()
    files = [os.path.join(npy_folder, file) for file in file_names]

    n = np.load(files[0]).shape[0]
    data_matrix = np.zeros((n, n, len(files)))
    final_matrix = np.zeros((n, n))

    final_matrix_G1 = np.zeros((n, n))
    final_matrix_S = np.zeros((n, n))
    final_matrix_G2 = np.zeros((n, n))


    for i, file in enumerate(files):
        m = np.load(file)
        data_matrix[:, :, i] = m / m.sum()

    n_row = 6
    fig, ax = plt.subplots(n_row, n_row)

    for i in range(n_row):
        for j in range(n_row):
            data_vector = data_matrix[i, j, :]
            ax[i][j].plot(data_vector)
            ax[i][j].xaxis.set_visible(False)
            ax[i][j].yaxis.set_visible(False)

    plt.savefig("./figures/trends_{}.png".format(npy_folder.split("_")[-2]))
    plt.close()

    mean_diagonal_signal = [np.sum([np.mean(data_matrix[i, i+d, :]) for i in range(n-d)])+0.0001 for d in range(n)]

    for i in range(n):
        for j in range(i, n):
            # DIFF BETWEEN S and G1+G2
            # data_vector = data_matrix[i, j, :]
            # signal_diff = np.mean(list(data_vector[:n//4])+list(data_vector[3*n//4:])) - data_vector[n//4:3*n//4].mean()
            # final_matrix[i, j] = final_matrix[j, i] = signal_diff/data_vector.mean()

            m_diag_signal = mean_diagonal_signal[np.abs(i-j)]
            data_vector_G1 = data_matrix[i, j, :25]
            data_vector_S = data_matrix[i, j, 25:67]
            data_vector_G2 = data_matrix[i, j, 67:]

            reg = LinearRegression().fit(np.array(range(len(data_vector_G1))).reshape(-1, 1), data_vector_G1)
            final_matrix_G1[i, j] = final_matrix_G1[j, i] = reg.coef_[0]/m_diag_signal
            reg = LinearRegression().fit(np.array(range(len(data_vector_S))).reshape(-1, 1), data_vector_S)
            final_matrix_S[i, j] = final_matrix_S[j, i] = reg.coef_[0]/m_diag_signal
            reg = LinearRegression().fit(np.array(range(len(data_vector_G2))).reshape(-1, 1), data_vector_G2)
            final_matrix_G2[i, j] = final_matrix_G2[j, i] = reg.coef_[0]/m_diag_signal
            # print(final_matrix_G2[i, j], final_matrix_G2[j, i], reg.coef_[0])
    # DIFF BETWEEN S and G1+G2
    # plot = plt.imshow(final_matrix, cmap='bwr', vmin=-0.75, vmax=0.75)
    # plt.title(npy_folder.split("_")[-2])
    # cbar = plt.colorbar(plot)
    # cbar.ax.get_yaxis().set_ticks([])
    # cbar.ax.text(3.5, 0.45, "Higher signal in G1/G2", ha='center', va='center')
    # cbar.ax.text(3.5, -0.45, "Higher signal in S", ha='center', va='center')
    # plt.savefig("./figures/trend_matrix_{}_{}.png".format(npy_folder.split("_")[-2], str(n).zfill(4)))
    # plt.close()

    t_val = 5e-4
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    sns.heatmap(final_matrix_G1, cmap='bwr', vmin=-t_val, vmax=t_val, ax=ax[0], square=True, cbar=False)
    sns.heatmap(final_matrix_S, cmap='bwr', vmin=-t_val, vmax=t_val, ax=ax[1], square=True, cbar=False)
    sns.heatmap(final_matrix_G2, cmap='bwr', vmin=-t_val, vmax=t_val, ax=ax[2], square=True, cbar=False)
    ax[0].set_title(npy_folder.split("_")[-2]+" phase G1")
    ax[1].set_title(npy_folder.split("_")[-2] + " phase S")
    ax[2].set_title(npy_folder.split("_")[-2] + " phase G2")
    fig.tight_layout()
    plt.savefig("./figures/trend_matrices_phases_{}_{}.png".format(npy_folder.split("_")[-2], str(n).zfill(4)))
    plt.close()


if __name__ == "__main__":
    path = "./examples/k562/npy_chr2_0050"

    # plot_trend_matrix(path)
