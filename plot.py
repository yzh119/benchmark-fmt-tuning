import matplotlib
import matplotlib.pyplot as plt
import dgl
import csv
import pickle
from dgl.data import PPIDataset, RedditDataset

plt.rcParams.update({'font.size': 5})

def get_data(filename):
    Xs = []
    coo_gflops = []
    csr_gflops = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            Xs.append(int(row[0]))
            coo_gflops.append(float(row[1]))
            csr_gflops.append(float(row[3]))
    return Xs, coo_gflops, csr_gflops

if __name__ == '__main__':
    fig, axs = plt.subplots(3, 6, figsize=(8, 3))

    # reddit
    reddit = RedditDataset()
    g = reddit.graph
    axs[0, 0].hist(g.in_degrees(), range=(0, 100))
    axs[0, 0].set_yticks([])
    for row, mode in enumerate(['spmm', 'sddmm']):
        filename = 'reddit_{}.csv'.format(mode)
        Xs, coo_gflops, csr_gflops = get_data(filename)
        axs[row + 1, 0].plot(Xs, coo_gflops, linewidth=1, label='coo')
        axs[row + 1, 0].plot(Xs, csr_gflops, linewidth=1, label='csr')
        axs[row + 1, 0].set_yticks([0, 200, 400])
        axs[row + 1, 0].set_xscale('log', basex=2)
        axs[row + 1, 0].set_xticks(Xs)
        axs[2, 0].set_xlabel('reddit')
    
    # mesh
    for i, k in enumerate([8, 16, 32, 64]):
        f = open('modelnet40_{}.g'.format(k), 'rb')
        gs = pickle.load(f)
        g = dgl.batch(gs)
        axs[0, i + 1].hist(g.in_degrees(), range=(0, 100))
        axs[0, i + 1].set_yticks([])
        for row, mode in enumerate(['spmm', 'sddmm']):
            filename = 'mesh_{}_{}.csv'.format(k, mode)
            Xs, coo_gflops, csr_gflops = get_data(filename)
            axs[row + 1, i + 1].plot(Xs, coo_gflops, linewidth=1, label='coo')
            axs[row + 1, i + 1].plot(Xs, csr_gflops, linewidth=1, label='csr')
            axs[row + 1, i + 1].set_yticks([0, 200, 400])
            axs[row + 1, i + 1].set_xscale('log', basex=2)
            axs[row + 1, i + 1].set_xticks(Xs)
            axs[2, i + 1].set_xlabel('mesh {}'.format(k))
    # ppi
    ppi = PPIDataset('train')
    g = ppi.graph
    axs[0, 5].hist(g.in_degrees(), range=(0, 100))
    axs[0, 5].set_yticks([])
    for row, mode in enumerate(['spmm', 'sddmm']):
        filename = 'ppi_{}.csv'.format(mode, k)
        Xs, coo_gflops, csr_gflops = get_data(filename)
        axs[row + 1, 5].plot(Xs, coo_gflops, linewidth=1, label='coo')
        axs[row + 1, 5].plot(Xs, csr_gflops, linewidth=1, label='csr')
        axs[row + 1, 5].set_yticks([0, 200, 400])
        axs[row + 1, 5].set_xscale('log', basex=2)
        axs[row + 1, 5].set_xticks(Xs)
    axs[2, 5].set_xlabel('ppi'.format(k))

    axs[0, 0].yaxis.set_label_position("left")
    axs[0, 0].set_ylabel('degree distribution')
    axs[1, 0].set_ylabel('GFLOPS(SPMM)')
    axs[2, 0].set_ylabel('GFLOPS(SDDMM)')

    fig.tight_layout()
    fig.savefig('1.pdf')
