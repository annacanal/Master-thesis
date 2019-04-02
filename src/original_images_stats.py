import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind
import os
import matplotlib.pyplot as plt
import spm1d
import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from statistics import mean
import get_compressed_data


def Pearson_corr(d1,d2):
    r, p = stats.pearsonr(d1, d2)
    return r,p

def buildPearsonSimilarityMatrix(data):
    numOfSamples = len(data)
    matrix = np.zeros(shape=(numOfSamples, numOfSamples))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            r, p = stats.pearsonr(data[i], data[j])
            matrix[i,j] = r
    return matrix

def main():
    data_pathMCI = "../Data/MCI_data"
    data_pathAD = "../Data/AD_data"
    data_pathCN = "../Data/CN_data"
    fileMCI = "mci.txt"
    fileAD = "ad.txt"
    fileCN = "cn.txt"
    n_mci= 522
    n_ad= 243
    n_cn = 304
    IM_X = 80
    IM_Y = 80
    IM_Z = 80

    images_MCI,labels_MCI,names_MCI = get_compressed_data.read_images_group(data_pathMCI,fileMCI,n_mci,"MCI")
    images_AD,labels_AD,names_AD = get_compressed_data.read_images_group(data_pathAD,fileAD, n_ad,"AD")
    images_CN,labels_CN,names_CN = get_compressed_data.read_images_group(data_pathCN,fileCN,n_cn,"CN")

    ######## Cut images to 80x80x80
    data_MCI = np.zeros((len(images_MCI), IM_X, IM_Y, IM_Z))
    data_AD = np.zeros((len(images_AD), IM_X, IM_Y, IM_Z))
    data_CN = np.zeros((len(images_CN), IM_X, IM_Y, IM_Z))
    for i, element in enumerate(images_MCI):
        data_MCI[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_AD):
        data_AD[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_CN):
        data_CN[i] = element[5:85, 15:95, 5:85]

    ######### Flatten the 3D images
    d_MCI = data_MCI.reshape(len(images_MCI), IM_X * IM_Y * IM_Z)
    d_AD = data_AD.reshape(len(images_AD), IM_X * IM_Y * IM_Z)
    d_CN = data_CN.reshape(len(images_CN), IM_X * IM_Y * IM_Z)
    data = np.concatenate((d_MCI, d_AD, d_CN))

    ######### CORRELATION MATRIX
    pearson_sim_corr = buildPearsonSimilarityMatrix(data)
    plt.matshow(pearson_sim_corr)
    plt.title("MCI-AD-CN pearson similarity matrix original images")
    plt.margins(0.2)
    plt.colorbar()
    plt.savefig('pearson similarity matrix original images.png')
    plt.show()




if __name__ == "__main__":
    main()
