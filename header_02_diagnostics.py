import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import seaborn as sns
import pandas as pd
import numpy as np


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
        for j in range(i+1, model.shape[0]):
            dist_pre.append(pre_dist_mat[i, j])
            dist_post.append(post_dist_mat[i, j])
            if i//size == j//size:
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
