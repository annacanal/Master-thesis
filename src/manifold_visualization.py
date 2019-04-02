import os
import numpy as np
import tSNE_visualization
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import preprocessing
import analysis_results
import tsne_original
import tSNE_dim_reduction


def main():
    data_CAE_80 = "../Results/CAE_3D_80"
    data_CAE_40 = "../Results/CAE_3D_40"
    data_path_tSNE = "../Results/tSNE"
    data_path_PCA = "../Results/PCA"   
    data_path_PCA_80_slices = "../Results/PCA/80_groups_slices"
    data_path_PCA_groups= "../Results/PCA/groups"
    CAE_slices = "../Results/CAE_results/Slices"
    CAE= "../Results/CAE_results"

    datapath_results = "../Results/tSNE/visualization"

    ### CAE SLICES 50 epochs
    labels_CAE_slice1 = np.load(os.path.join(CAE_slices, "labels_slices1.npy"))
    labels_CAE_slice2 = np.load(os.path.join(CAE_slices, "labels_slices2.npy"))
    labels_CAE_slice3 = np.load(os.path.join(CAE_slices, "labels_slices3.npy"))
    CAEs1_128nodes = np.load(os.path.join(CAE_slices, "s1_128nodes_120epochs.npy"))
    CAEs2_128nodes = np.load(os.path.join(CAE_slices, "s2_128nodes_120epochs.npy"))
    CAEs3_128nodes = np.load(os.path.join(CAE_slices, "s3_128nodes_120epochs.npy"))
    CAEs1_512nodes = np.load(os.path.join(CAE_slices, "s1_512nodes_120epochs.npy"))
    CAEs2_512nodes = np.load(os.path.join(CAE_slices, "s2_512nodes_120epochs.npy"))
    CAEs3_512nodes = np.load(os.path.join(CAE_slices, "s3_512nodes_120epochs.npy"))
    labels_CAE= np.load(os.path.join(CAE, "labels_test_80.npy"))
    CAE_128nodes = np.load(os.path.join(CAE, "cae80_128nodes_40epochs.npy"))
    CAE_5500nodes = np.load(os.path.join(CAE, "cae80_5500nodes_40epochs.npy"))

    # Load Data
    # Load compressed data from 3D CAE with images of 80x80x80 (CAE80) and 40x80x40 (CAE40)
    # labels_CAE_80 = np.load(os.path.join(data_CAE_80, "labels_test_80.npy"))
    # labels_CAE_40 = np.load(os.path.join(data_CAE_40, "labels_test_40.npy"))
    # compressed80_128nodes_10_lr_0_001 = np.load(os.path.join(data_CAE_80, "compressed80_128nodes_10_lr0.001.npy"))
    # compressed80_128nodes_10_lr_0_1 = np.load(os.path.join(data_CAE_80, "compressed80_128nodes_10_lr0.1.npy"))
    # compressed80_5500nodes_10_lr_0_001 = np.load(os.path.join(data_CAE_80, "compressed80_5500nodes_10_lr0.001.npy"))
    # compressed80_5500nodes_10_lr_0_1 = np.load(os.path.join(data_CAE_80, "compressed80_5500nodes_10_lr0.1.npy"))
    # compressed40_128nodes_10_lr_0_001 = np.load(os.path.join(data_CAE_40, "compressed40_128nodes_10_lr0.001.npy"))
    # compressed40_128nodes_10_lr_0_1 = np.load(os.path.join(data_CAE_40, "compressed40_128nodes_10_lr0.1.npy"))
    # compressed40_1500nodes_10_lr_0_001 = np.load(os.path.join(data_CAE_40, "compressed40_1500nodes_10_lr0.001.npy"))
    # compressed40_1500nodes_10_lr_0_1 = np.load(os.path.join(data_CAE_40, "compressed40_1500nodes_10_lr0.1.npy"))

    
    ## SELECT DATA
    data = CAE_5500nodes
    filename= "CAE_5500nodes"
    labels = labels_CAE

    # Reduce dimensionality
    l_rates = [500, 650]    # l_rates = [200,350, 500, 650, 800]
    exaggeration = [50, 80]#[12, 50,80]
    ##TSNE MANIDOLF VISUALIZATION
    for l_rate in l_rates:
        for ex in exaggeration:
            embedded1,tsne = tSNE_visualization.reduction(data,perplexity=20, l_r = l_rate, dim = 2, ex = ex)
            tSNE_visualization.plot_clusters(embedded1, labels, filename+"_l.rate-"+str(l_rate)+"_ex-"+str(ex), datapath_results, type="actual")

if __name__ == "__main__":
    main()


