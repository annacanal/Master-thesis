import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import preprocessing
import get_compressed_data
import random

IM_X = 80
IM_Y = 80
IM_Z = 80

def reduction_sklearn(data, perplexity=30.0, l_r = 200.0, dim = 2, ex = 12.0, iterations = 5000, metric='euclidean',init='random', verbosity = 0):
    tsne = TSNE(n_components= dim, perplexity=perplexity, early_exaggeration=ex, learning_rate=l_r, n_iter=iterations,n_iter_without_progress=300,
                min_grad_norm=1e-07, metric=metric, init =init, verbose = 0, random_state = None, method ='barnes_hut', angle = 0.5)
    embedded_data = tsne.fit_transform(data)
    return embedded_data, tsne

def main():
    data_pathMCI = "../Data/MCI_data"
    data_pathAD = "../Data/AD_data"
    data_pathCN = "../Data/CN_data"
    fileMCI = "mci.txt"
    fileAD = "ad.txt"
    fileCN = "cn.txt"
    n_mci= 522 #522
    n_ad= 243  #243
    n_cn = 304 #304
    # datapath_results = "../Results/tSNE"

    #Load Data
    images,labels, names= get_compressed_data.read_images(data_pathMCI,data_pathAD,data_pathCN, fileMCI, fileAD, fileCN, n_mci, n_ad, n_cn)
    # Cut images in [80,80,80], save them in array data
    data = np.zeros((len(images), IM_X, IM_Y, IM_Z))
    for i, element in enumerate(images):
        data[i] = element[5:85, 15:95, 5:85]
    data = data.astype('float32') / 255
    # Standardization
    data = data.reshape(len(images), IM_X * IM_Y * IM_Z)
    data = preprocessing.scale_select(data)

    # Shuffle data
    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(labels)

    # Reduce dimensionality
    l_rates = [500, 650]
    exaggeration = [ 12, 50, 80]
    embedded_data = []
    param_embedded_data = []

    for l_rate in l_rates:
        for ex in exaggeration:
            embedded,tsne = reduction_sklearn(data,perplexity=20, l_r = l_rate, dim = 2, ex = ex, iterations = 5000, metric='euclidean' )
            embedded_data.append(embedded)
            param_embedded_data.append([l_rate,ex])

    # save data
    np.save("embedded_data",embedded_data)
    np.save("param_embedded_data",param_embedded_data)
    np.save("labels",labels)



if __name__ == "__main__":
    main()

