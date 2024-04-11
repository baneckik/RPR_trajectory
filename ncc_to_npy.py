import numpy as np
import pandas as pd
import os
from header_00_matrix import matrix_plot
from tqdm import tqdm
from header_02_diagnostics import plot_trend_matrix
import multiprocessing as mp


def ncc_to_npy(path, chrom, size=100, binary=True):
    df = pd.read_csv(path, header=None, sep="\t")
    df = df[[0, 2, 6, 8]]
    df.columns = ["chr1", "pos1", "chr2", "pos2"]
    df = df[(df.chr1 == chrom) & (df.chr2 == chrom)]

    pos_min = np.min(list(df.pos1) + list(df.pos2))
    pos_max = np.max(list(df.pos1) + list(df.pos2))

    array = np.zeros((size, size))
    for i in range(df.shape[0]):
        pos1_i = int((df.pos1.iloc[i] - pos_min) / (pos_max - pos_min + 1) * size)
        pos2_i = int((df.pos2.iloc[i] - pos_min) / (pos_max - pos_min + 1) * size)
        array[pos1_i, pos2_i] += 1
        array[pos2_i, pos1_i] += 1

    if binary:
        return np.where(array > 0.5, 1, 0)
    else:
        return array


def ncc_folder_to_npy(input_folder, output_folder, chrom, size, png_folder=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = os.listdir(input_folder)
    files = [file for file in files if file.endswith(".ncc")]
    for file in tqdm(files):
        file_path = os.path.join(input_folder, file)
        path_out = os.path.join(output_folder, file[:-4] + "_{}.npy".format(chrom))
        hic_array = ncc_to_npy(file_path, chrom, size, binary=False)
        if png_folder is not None:
            matrix_plot(hic_array, os.path.join(png_folder, "schic.png"))
        np.save(path_out, hic_array)


def parse_chroms(args):
    chroms, resolutions = args
    for chrom in chroms:
        for matrix_size in resolutions:
            main_path = "./examples/k562"
            chromosome = "chr" + str(chrom)

            ncc_path = main_path + "/ncc"
            npy_path = main_path + "_".join(["/npy", chromosome, str(matrix_size).zfill(4)])

            ncc_folder_to_npy(ncc_path, npy_path, chromosome, matrix_size, main_path)

            path = "./examples/k562/npy_{}_{}".format(chromosome, str(matrix_size).zfill(4))
            plot_trend_matrix(path)


if __name__ == "__main__":
    # main_path = "./examples/k562"
    # chromosome = "chr2"
    # matrix_size = 50
    #
    # ncc_path = main_path + "/ncc"
    # npy_path = main_path + "_".join(["/npy", chromosome, str(matrix_size).zfill(4)])
    #
    # ncc_folder_to_npy(ncc_path, npy_path, chromosome, matrix_size, main_path)

    chroms = list(range(1, 23))+["X"]
    resolutions = [50, 100]

    threads = 22
    task_list = [(chroms_thread, resolutions)
                 for chroms_thread in np.array_split(chroms, threads)]

    pool = mp.Pool(threads)
    results = pool.map(parse_chroms, task_list)
    pool.close()
    pool.join()


