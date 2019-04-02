import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import preprocessing
import get_compressed_data
import random

IM_X = 80
IM_Y = 80
IM_Z= 80

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
    # Cut images in slices of [80,80], save them in array data
    data_s1 = np.zeros((len(images),IM_X, IM_Y))
    data_s2 = np.zeros((len(images), IM_X, IM_Z))
    data_s3 = np.zeros((len(images), IM_Y, IM_Z))
    for i, element in enumerate(images):
        data_s1[i] = element[5:85, 15:95, 45]
        data_s2[i] = element[5:85, 55, 5:85, ]
        data_s3[i] = element[45, 15:95, 5:85, ]
    data_s1 = data_s1.astype('float32') / 4095
    data_s2 = data_s2.astype('float32') / 4095
    data_s3 = data_s3.astype('float32') / 4095
    # Standardization
    data_s1 = data_s1.reshape(len(images), IM_X * IM_Y)
    data_s1 = preprocessing.scale_select(data_s1)
    data_s2 = data_s2.reshape(len(images), IM_X * IM_Z)
    data_s2 = preprocessing.scale_select(data_s2)
    data_s3 = data_s3.reshape(len(images), IM_Y * IM_Z)
    data_s3 = preprocessing.scale_select(data_s3)

    # Shuffle data
    random.seed(42)
    random.shuffle(data_s1)
    random.seed(42)
    random.shuffle(data_s2)
    random.seed(42)
    random.shuffle(data_s3)
    random.seed(42)
    random.shuffle(labels)

    # Reduce dimensionality
    l_rates = [500, 650]
    exaggeration = [ 12, 50, 80]
    embedded_datas1 = []
    param_embedded_datas1 = []
    embedded_datas2 = []
    param_embedded_datas2 = []
    embedded_datas3 = []
    param_embedded_datas3 = []

    for l_rate in l_rates:
        for ex in exaggeration:
            embeddeds1,tsne = reduction_sklearn(data_s1,perplexity=20, l_r = l_rate, dim = 2, ex = ex, iterations = 5000, metric='euclidean' )
            embedded_datas1.append(embeddeds1)
            param_embedded_datas1.append([l_rate,ex])
            embeddeds2,tsne = reduction_sklearn(data_s2,perplexity=20, l_r = l_rate, dim = 2, ex = ex, iterations = 5000, metric='euclidean' )
            embedded_datas2.append(embeddeds2)
            param_embedded_datas2.append([l_rate,ex])
            embeddeds3,tsne = reduction_sklearn(data_s3,perplexity=20, l_r = l_rate, dim = 2, ex = ex, iterations = 5000, metric='euclidean' )
            embedded_datas3.append(embeddeds3)
            param_embedded_datas3.append([l_rate,ex])

    # save data
    np.save("embedded_datas1",embedded_datas1)
    np.save("param_embedded_datas1",param_embedded_datas1)
    np.save("embedded_datas2",embedded_datas2)
    np.save("param_embedded_datas2",param_embedded_datas2)
    np.save("embedded_datas3",embedded_datas3)
    np.save("param_embedded_datas3",param_embedded_datas3)
    np.save("labels_tSNE_2",labels)



if __name__ == "__main__":
    main()
