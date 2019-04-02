import numpy as np
from sklearn.decomposition import PCA
import preprocessing
import get_compressed_data
import random

IM_X = 40
IM_Y = 80
IM_Z= 40

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

    #### SELECT GROUP #####
    type = "ALL"
    data_path = data_pathCN
    filename = fileCN
    n = n_cn


    #Load Data
    images,labels, names= get_compressed_data.read_images(data_pathMCI,data_pathAD,data_pathCN, fileMCI, fileAD, fileCN, n_mci, n_ad, n_cn)
    #images,labels, names= get_compressed_data.read_images_group(data_path, filename, n, type)
    # Cut images in slices of [80,80], save them in array data
    data_s1 = np.zeros((len(images),IM_X, IM_Y))
    data_s2 = np.zeros((len(images), IM_X, IM_Z))
    data_s3 = np.zeros((len(images), IM_Y, IM_Z))
    for i, element in enumerate(images):
        data_s1[i] = element[25:65, 20:100, 45] # 40x80x40
        data_s2[i] = element[25:65, 60, 25:65] # 40x80x40
        data_s3[i] = element[45, 20:100, 25:65] # 40x80x40
        # data_s1[i] = element[5:85, 15:95, 45] # 80x80x80
        # data_s2[i] = element[5:85, 55, 5:85] # 80x80x80
        # data_s3[i] = element[45, 15:95, 5:85] # 80x80x80
    #data_s1 = data_s1 / 4095
    #data_s2 = data_s2 / 4095
    #data_s3 = data_s3 / 4095
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

    print(data_s1.shape)
    print(data_s2.shape)
    ##### Reduce dimensionality
    print("Reduce dimensionality with PCA")
    dimensions = [1024,128,50] # 512000 is the maximum
    for dim in dimensions:
        embeddeds1,tsne = reduction(data_s1, dim )
        np.save(type+"small_datas1_pca"+str(dim), embeddeds1)
        embeddeds2,tsne = reduction(data_s2, dim )
        np.save(type+"small_datas2_pca"+str(dim), embeddeds2)
        embeddeds3,tsne = reduction(data_s3, dim )
        np.save(type+"small_datas3_pca"+str(dim), embeddeds3)
    # save data
    np.save(type+"small_labels_pca_slices",labels)



if __name__ == "__main__":
    main()
