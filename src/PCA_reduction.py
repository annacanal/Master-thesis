import numpy as np
from sklearn.decomposition import PCA
import preprocessing
import get_compressed_data
import random

IM_X = 80
IM_Y = 80
IM_Z = 80


def reduction(data, dim = 2,  svd_solver = 'auto'):
    pca = PCA(n_components = dim, svd_solver = svd_solver)
    embedded_data_pca = pca.fit_transform(data)
    return embedded_data_pca, pca


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
    print("Reduce dimensionality with PCA")
    dimensions = [10000,1000,100,2] # 512000 is the maximum
    for dim in dimensions:
        embedded,tsne = reduction(data, dim )
        np.save("embedded_data_pca_"+str(dim), embedded)
    # save data
    np.save("labels",labels)



if __name__ == "__main__":
    main()
