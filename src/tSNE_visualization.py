from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

def plot_clusters(samples, targets, title, datapath, type="predicted"):
    clustered = [[] for _ in np.unique(targets)]

    for i in range(len(samples)):
        clustered[int(targets[i])].append(samples[i])

    if type == "actual":
        labels = ["MCI", "CN", "AD"]
        colors = ["red", "green", "blue"]
        cmap = "seismic"
    if type == "predicted":
        labels = ["cluster_{}".format(i) for i in np.unique(targets)]
        colors = []
        cmap = "RdBu"

    markers = [".", "*", "+"]

    for i, cluster in enumerate(clustered):
        cluster = np.array(cluster)
        if type == "actual":
            plt.scatter(cluster[:, 0], cluster[:, 1], cmap=cmap, marker=markers[i], label=labels[i])
        elif type == "predicted":
            plt.scatter(cluster[:, 0], cluster[:, 1], label=labels[i], cmap=cmap, marker='.', linewidths=0)

    plt.legend()
    plt.title("{} clusters".format(title))
    output_filename = "{}_clusters.png".format("_".join(title.split()))
    plt.savefig(os.path.join(datapath, output_filename))
    plt.show()

def reduction(data, perplexity, l_r = 200, dim = 2, ex = 12, iterations = 5000, verbosity = 0):
    tsne = TSNE(n_components = dim, n_jobs = -1, learning_rate = l_r,
                            perplexity=perplexity, early_exaggeration = ex,
                            n_iter = iterations, random_state= 42,
                            verbose = verbosity)

    reduced_samples = tsne.fit_transform(data)

    return reduced_samples, tsne
