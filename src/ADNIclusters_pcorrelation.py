import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import spm1d
import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from statistics import mean

def class_to_numerical(target):
    if target == "mci_stable":
        return 0
    elif target == "progressive1":
        return 1
    elif target == "progressive2":
        return 2
    elif target == "progressive3":
        return 3
    elif target == "progressive4":
        return 4
    elif target == "progressive5":
        return 5
    else:
        raise ValueError("Unexpected target value: {}".format(target))

def read_images_group(data_path, filename, type):
    images = []
    names= []
    labels = []
    ######## MCI ########
    with open(os.path.join(data_path, filename), "r") as f:
        files = f.readlines() # Read the whole file at once
    files = [x.strip() for x in files]
    #for i in files1:
    for i in files:
        names.append(i)
        im = np.load(os.path.join(data_path,i))
        images.append(im['arr_0'])
        group = class_to_numerical(type)
        labels.append(group) # group =0 = MCI, group = 1 =AD, group=2 CN

    return images, labels, names

def Pearson_corr(d1,d2):
    r, p = stats.pearsonr(d1, d2)
    return r,p

def WithinGroup_analysis(data):
    r_scores = []
    p_values = []
    values = []
    for i in range(len(data) - 1):
        # Perform Pearson Correlation
        r, p3 = Pearson_corr(data[i], data[i + 1])
        r_scores.append(r)
        p_values.append(p3)
    values.append(np.mean(r_scores))
    values.append(np.mean(p_values))
    return values

def BetweenGroups_analysis(data1, data2):
    r_scores = []
    p_values = []
    values = []
    for el1 in data1:
        for el2 in data2:
            # Perform Pearson Correlation
            r, p3 = Pearson_corr(el1, el2)
            r_scores.append(r)
            p_values.append(p3)
    values.append(np.mean(r_scores))
    values.append(np.mean(p_values))
    return values

def buildPearsonSimilarityMatrix(stable_values, p1_values, p2_values, p3_values, p4_values,p5_values, stable_p1_values,stable_p2_values,stable_p3_values,stable_p4_values,stable_p5_values,p1_p2_values,p1_p3_values,
                                 p1_p4_values,p1_p5_values,p2_p3_values,p2_p4_values,p2_p5_values,p3_p4_values,p3_p5_values,p4_p5_values):
    matrix= [[stable_values[0],stable_p1_values[0],stable_p2_values[0],stable_p3_values[0],stable_p4_values[0],stable_p5_values[0]],
             [stable_p1_values[0],p1_values[0],p1_p2_values[0],p1_p3_values[0],p1_p4_values[0],p1_p5_values[0]],
             [stable_p2_values[0],p1_p2_values[0],p2_values[0],p2_p3_values[0],p2_p4_values[0],p2_p5_values[0]],
             [stable_p3_values[0],p1_p3_values[0],p2_p3_values[0],p3_values[0],p3_p4_values[0],p3_p5_values[0]],
             [stable_p4_values[0],p1_p4_values[0],p2_p4_values[0],p3_p4_values[0],p4_values[0],p4_p5_values[0]],
             [stable_p5_values[0],p1_p5_values[0],p2_p5_values[0],p3_p5_values[0],p4_p5_values[0],p5_values[0]]]
    return matrix

def main():
    data_path_stable = "../Data/MCI_data_clustered/mci_stable"
    data_path_p1 = "../Data/MCI_data_clustered/progressive1"
    data_path_p2 = "../Data/MCI_data_clustered/progressive2"
    data_path_p3 = "../Data/MCI_data_clustered/progressive3"
    data_path_p4 = "../Data/MCI_data_clustered/progressive4"
    data_path_p5 = "../Data/MCI_data_clustered/progressive5"
    file0 = "mci_stable.txt"
    file1 = "progressive1.txt"
    file2 = "progressive2.txt"
    file3 = "progressive3.txt"
    file4 = "progressive4.txt"
    file5 = "progressive5.txt"
    IM_X = 80
    IM_Y = 80
    IM_Z = 80
    data_filename = 'ADNI_clusters_within.html'  # aquest és el valid

    images_stable,labels_stable,names_stable = read_images_group(data_path_stable,file0,"mci_stable")
    images_p1,labels_p1,names_p1 = read_images_group(data_path_p1,file1,"progressive1")
    images_p2,labels_p2,names_p2 = read_images_group(data_path_p2,file2,"progressive2")
    images_p3,labels_p3,names_p3 = read_images_group(data_path_p3,file3,"progressive3")
    images_p4,labels_p4,names_p4 = read_images_group(data_path_p4,file4,"progressive4")
    images_p5, labels_p5, names_p5 = read_images_group(data_path_p5, file5, "progressive5")

    ######## Cut images to 80x80x80
    data_stable = np.zeros((len(images_stable), IM_X, IM_Y, IM_Z))
    data_p1 = np.zeros((len(images_p1), IM_X, IM_Y, IM_Z))
    data_p2 = np.zeros((len(images_p2), IM_X, IM_Y, IM_Z))
    data_p3 = np.zeros((len(images_p3), IM_X, IM_Y, IM_Z))
    data_p4 = np.zeros((len(images_p4), IM_X, IM_Y, IM_Z))
    data_p5 = np.zeros((len(images_p5), IM_X, IM_Y, IM_Z))
    for i, element in enumerate(images_stable):
        data_stable[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_p1):
        data_p1[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_p2):
        data_p2[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_p3):
        data_p3[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_p4):
        data_p4[i] = element[5:85, 15:95, 5:85]
    for i, element in enumerate(images_p5):
        data_p5[i] = element[5:85, 15:95, 5:85]
    ######### Flatten the 3D images
    d_stable = data_stable.reshape(len(images_stable), IM_X * IM_Y * IM_Z)
    d_p1 = data_p1.reshape(len(images_p1), IM_X * IM_Y * IM_Z)
    d_p2 = data_p2.reshape(len(images_p2), IM_X * IM_Y * IM_Z)
    d_p3 = data_p3.reshape(len(images_p3), IM_X * IM_Y * IM_Z)
    d_p4 = data_p4.reshape(len(images_p4), IM_X * IM_Y * IM_Z)
    d_p5 = data_p5.reshape(len(images_p5), IM_X * IM_Y * IM_Z)

    ##### PEARSON CORRELATION ########3
    # Statistics within Group mci stable
    stable_values = WithinGroup_analysis(d_stable)
    # Statistics within Group progressive 1
    p1_values = WithinGroup_analysis(d_p1)
    # Statistics within Group progressive 2
    p2_values = WithinGroup_analysis(d_p2)
    # Statistics within Group progressive 3
    p3_values = WithinGroup_analysis(d_p3)
    # Statistics within Group progressive 4
    p4_values = WithinGroup_analysis(d_p4)
    # Statistics within Group progressive 5
    p5_values = WithinGroup_analysis(d_p5)

    # Statistics between stable i p1
    stable_p1_values = BetweenGroups_analysis(d_stable, d_p1)
    # Statistics between stable i p2
    stable_p2_values = BetweenGroups_analysis(d_stable, d_p2)
    # Statistics between stable i p1
    stable_p3_values = BetweenGroups_analysis(d_stable, d_p3)
    # Statistics between stable i p1
    stable_p4_values = BetweenGroups_analysis(d_stable, d_p4)
    # Statistics between stable i p1
    stable_p5_values = BetweenGroups_analysis(d_stable, d_p5)
    # Statistics between p1 i p2
    p1_p2_values = BetweenGroups_analysis(d_p1, d_p2)
    # Statistics between p1 i p3
    p1_p3_values = BetweenGroups_analysis(d_p1, d_p3)
    # Statistics between p1 i p4
    p1_p4_values = BetweenGroups_analysis(d_p1, d_p4)
    # Statistics between p1 i p5
    p1_p5_values = BetweenGroups_analysis(d_p1, d_p5)
    # Statistics between p2 i p3
    p2_p3_values = BetweenGroups_analysis(d_p2, d_p3)
    # Statistics between p2 i p4
    p2_p4_values = BetweenGroups_analysis(d_p2, d_p4)
    # Statistics between p2 i p5
    p2_p5_values = BetweenGroups_analysis(d_p2, d_p5)
    # Statistics between p3 i p4
    p3_p4_values = BetweenGroups_analysis(d_p3, d_p4)
    # Statistics between p3 i p5
    p3_p5_values = BetweenGroups_analysis(d_p3, d_p5)
    # Statistics between p4 i p5
    p4_p5_values = BetweenGroups_analysis(d_p4, d_p5)
    
  
    ######### CORRELATION MATRIX
    labels = ["stable", "progressive 1", "progressive 2", "progressive 3", "progressive 4", "progressive 5"]
    pearson_sim_corr = buildPearsonSimilarityMatrix(stable_values, p1_values, p2_values, p3_values, p4_values,p5_values, stable_p1_values,stable_p2_values,stable_p3_values,stable_p4_values,stable_p5_values,p1_p2_values,p1_p3_values,
                                 p1_p4_values,p1_p5_values,p2_p3_values,p2_p4_values,p2_p5_values,p3_p4_values,p3_p5_values,p4_p5_values)
    plt.matshow(pearson_sim_corr)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.yticks(range(len(labels)), labels)
    plt.margins(0.2)
    plt.colorbar()
    plt.savefig('pearson similarity matrix MCI ADNI clusters.png')
    plt.show()






if __name__ == "__main__":
    main()
