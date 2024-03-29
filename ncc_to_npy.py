import numpy as np
import pandas as pd
import os
from header_00_matrix import matrix_plot


def ncc_to_npy(path, chrom, size=100, binary=True):
    df = pd.read_csv(path, header=None, sep=" ")
    df = df[[0, 2, 6, 8]]
    df.columns = ["chr1", "pos1", "chr2", "pos2"]
    df = df[(df.chr1 == chrom) & (df.chr2 == chrom)]

    pos_min = np.min(list(df.pos1)+list(df.pos2))
    pos_max = np.max(list(df.pos1) + list(df.pos2))

    array = np.zeros((size, size))
    for i in range(df.shape[0]):
        pos1_i = int((df.pos1.iloc[i]-pos_min)/(pos_max-pos_min+1)*size)
        pos2_i = int((df.pos2.iloc[i] - pos_min) / (pos_max - pos_min + 1) * size)
        array[pos1_i, pos2_i] += 1
        array[pos2_i, pos1_i] += 1

    if binary:
        return np.where(array > 0.5, 1, 0)
    else:
        return array


if __name__ == "__main__":
    ncc_path = "./examples"
    chrom = "chr1"
    size = 200

    files = os.listdir(ncc_path)
    for file in files:
        file_path = os.path.join(ncc_path, file)
        path_out = file_path[:-4] + "_{}.npy".format(chrom)
        hic_array = ncc_to_npy(file_path, chrom, size)
        matrix_plot(hic_array, "./test_schic.png")
        np.save(path_out, hic_array)


