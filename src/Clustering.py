from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization.python.ops import clustering_ops
import os
from matplotlib import pyplot as plt
import preprocessing
import tSNE_visualization
import analysis_results
import random

def fit_kmeans(reduced_samples, n_clusters=3):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(reduced_samples)
    clusters = kmeans.labels_
    return clusters

def fit_DBSCAN(reduced_samples, eps, min_samples, metric = 'euclidean'):
    dbscan = DBSCAN(eps = eps, min_samples = min_samples,metric=metric) # metrics in sklearn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
    dbscan.fit(reduced_samples)
    clusters = dbscan.labels_
    unique, counts = np.unique(clusters, return_counts = True)
    for i, cluster in enumerate(clusters):
        if cluster == -1:  #Noisy samples are given the label -1.
            clusters[i] = unique[-1] + 1

    return clusters

def fit_agglomerative(reduced_samples, n_clusters=3):
    agg =AgglomerativeClustering(affinity='euclidean',linkage='ward', n_clusters=n_clusters)
    agg.fit(reduced_samples)
    clusters = agg.labels_
    return clusters


def create_labels(data_path1, data_path2, data_path3, data_path4,data_path5,data_path6, f1,f2,f3,f4,f5,f6, names):
    labels = []
    ######## MCI stable ########
    with open(os.path.join(data_path1, f1), "r") as f:
        files1 = f.readlines() # Read the whole file at once
    files1 = [x.strip() for x in files1]
    ######## progressive 1 ########
    with open(os.path.join(data_path2,f2), "r") as f:
        files2 = f.readlines() # Read the whole file at once
    files2 = [x.strip() for x in files2]
    ######## progressive 2 ########
    with open(os.path.join(data_path3,f3), "r") as f:
        files3 = f.readlines() # Read the whole file at once
    files3 = [x.strip() for x in files3]
    ######## progressive 3 ########
    with open(os.path.join(data_path4,f4), "r") as f:
        files4 = f.readlines() # Read the whole file at once
    files4 = [x.strip() for x in files4]
    ######## progressive 4 ########
    with open(os.path.join(data_path5,f5), "r") as f:
        files5 = f.readlines() # Read the whole file at once
    files5 = [x.strip() for x in files5]
    ######## progressive 5 ########
    with open(os.path.join(data_path6,f6), "r") as f:
        files6 = f.readlines() # Read the whole file at once
    files6 = [x.strip() for x in files6]

    for i in names:
        t=0
        for j in files1:
            if i in j:
                labels.append(0)
                t=1
        for j in files2:
            if i in j:
                labels.append(1)
                t=1
        for j in files3:
            if i in j:
                labels.append(2)
                t=1
        for j in files4:
            if i in j:
                labels.append(3)
                t=1
        for j in files5:
            if i in j:
                labels.append(4)
                t=1
        for j in files6:
            if i in j:
                labels.append(5)
                t=1
        if t==0:
            labels.append(0)
    return labels


def main():
    PERPLEXITY = 30
    LEARNING_RATE = 600
    EXAGGERATION = 84
    data_CAE_80_MCI= "../Results/CAE_groups/MCI"
    data_path_stable = "../Data/MCI_data_clustered/mci_stable"
    data_path_p1 = "../Data/MCI_data_clustered/progressive1"
    data_path_p2 = "../Data/MCI_data_clustered/progressive2"
    data_path_p3 = "../Data/MCI_data_clustered/progressive3"
    data_path_p4 = "../Data/MCI_data_clustered/progressive4"
    data_path_p5 = "../Data/MCI_data_clustered/progressive5"
    f1 = "mci_stable.txt"
    f2 = "progressive1.txt"
    f3 = "progressive2.txt"
    f4 = "progressive3.txt"
    f5 = "progressive4.txt"
    f6 = "progressive5.txt"
    datapath_results = "../Results"

    ##################3 CAE 3D groups ###############
    MCI80_128nodes= np.load(os.path.join(data_CAE_80_MCI, "cae80_128nodes.npy"))
    MCI80_5500nodes= np.load(os.path.join(data_CAE_80_MCI, "cae80_5500nodes.npy"))
    names_test = np.load(os.path.join(data_CAE_80_MCI, "names_test_MCI.npy"))
    # Build labels based on the 6 progression clusters
    labels_test = create_labels(data_path_stable, data_path_p1, data_path_p2, data_path_p3,data_path_p4,data_path_p5, f1,f2,f3,f4,f5,f6, names_test)
    # Select 128 or 5500 nodes reduced data
    compressed = MCI80_128nodes
    print(len(compressed))
    print(len(labels_test))
    tSNE_visualization.plot_clusters(compressed, labels_test,"MCI80_5500nodes_adni_c.png", datapath_results, type="actual")


    clustering_type = "agglomerative" #Kmeans or DBSCAN or agglomerative
    clusters_list = []
    # CLUSTERING
    if clustering_type == "Kmeans":
        print("Kmeans clustering")
        unsupervised_model = tf.contrib.learn.KMeansClustering(6,  distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE, initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT)
        def train_input_fn():
            data = tf.constant(compressed, tf.float32)
            return (data, None)
        unsupervised_model.fit(input_fn=train_input_fn, steps=1000)
        clusters = unsupervised_model.predict(input_fn=train_input_fn)
        index = 0
        for i in clusters:
            current_cluster = i['cluster_idx']
            clusters_list.append(current_cluster)
            index = index + 1
        analysis_results.evaluate(labels_test, clusters_list)
        cnf = analysis_results.create_confusion_matrix(labels_test,clusters_list)
        analysis_results.plot_confusion_matrix(cnf, classes=['Stable', 'Progressive1','Progressive2', 'Progressive3', 'Progressive4','Progressive5'], normalize=False,
                              title='Normalized confusion matrix')
        plt.show()


        print("mean accuracy: "+ str(analysis_results.mean_accuracy(labels_test, clusters_list)))

    elif clustering_type == "DBSCAN":
        print("DBSCAN clustering")
        #EPS = [0.5, 1.0, 1.5]
        EPS = [1.5,2.8,3,5]
        #EPS = 0.25  # 2.8 # The maximum distance between two samples for them to be considered as in the same neighborhood.
        MIN_SAMPLES =  np.arange(5, 15, 5) #[1,3]
        for n_eps in EPS:
            for n_samples in MIN_SAMPLES:
                clusters = fit_DBSCAN(compressed, eps=n_eps, min_samples=n_samples, metric='euclidean')
                print("accuracy with EPS: " + str(n_eps) + " ,SAMPLES: " + str(n_samples))
                analysis_results.evaluate(labels_test, clusters)
                print("mean accuracy: " + str(analysis_results.mean_accuracy(labels_test, clusters)))

                cnf = analysis_results.create_confusion_matrix(labels_test, clusters)
                analysis_results.plot_confusion_matrix(cnf, classes=['Stable', 'Progressive1', 'Progressive2', 'Progressive3','Progressive4', 'Progressive5'],
                                                       normalize=False, title='Normalized confusion matrix')
                plt.show()
                tSNE_visualization.plot_clusters(compressed, clusters, "MCI80_5500nodes_adni_c_after_dbscan_eps_"+str(n_eps)+"_nsamples_"+str(n_samples)+".png",  datapath_results, type="predicted")
                plt.show()


    elif clustering_type == "agglomerative":
        print("agglomerative clustering")
        clusters = fit_agglomerative(compressed, n_clusters=6)
        print("accuracy agg: " )
        analysis_results.evaluate(labels_test, clusters)
        print("mean accuracy: " + str(analysis_results.mean_accuracy(labels_test, clusters)))
        cnf = analysis_results.create_confusion_matrix(labels_test, clusters)
        analysis_results.plot_confusion_matrix(cnf, classes=['Stable', 'Progressive1', 'Progressive2', 'Progressive3',
                                                             'Progressive4', 'Progressive5'], normalize=False,
                                               title='Normalized confusion matrix')
        plt.show()
        tSNE_visualization.plot_clusters(compressed, clusters, "MCI80_5500nodes_adni_c_after_agglomerative_pre3.png", datapath_results,
                                         type="predicted")
        plt.show()


if __name__ == "__main__":
    main()
