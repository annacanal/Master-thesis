import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind
import os
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from scipy.spatial import distance
#from sklearn import preprocessing
import preprocessing
import random

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

def calculateSpread(x,mean):
    spread = []
    for element in x:
        dist = distance.euclidean(element,mean)
        spread.append(dist)
    return spread

def calculateMaxRadius(x,mean):
    spread = []
    for element in x:
        dist = distance.euclidean(element,mean)
        spread.append(dist)
    return max(spread)

# Randomly select 10 points of the group
def select_10samples(data):
    # shuffle data
    random.shuffle(data)
    samples = random.sample(list(data),10)
    return samples

def interclass_distances(data_samples1, data_samples2):
    total_distance = 0
    for el1 in data_samples1:
        for el2 in data_samples2:
            dist = distance.euclidean(el1,el2)
            total_distance += dist
    return total_distance
def interclass_distances2(data_samples1, data_samples2):
    dist = distance.euclidean(data_samples1,data_samples2)
    return dist

def intraclass_distances(data_samples):
    distances = distance.pdist(data_samples, metric = "euclidean")
    total_distance = np.sum(distances)
    return total_distance

def total_intradistance(stable_samples,p1_samples,p2_samples,p3_samples,p4_samples,p5_samples):
    stable_interdist= intraclass_distances(np.asarray(stable_samples))
    p1_interdist= intraclass_distances(np.asarray(p1_samples))
    p2_interdist= intraclass_distances(np.asarray(p2_samples))
    p3_interdist= intraclass_distances(np.asarray(p3_samples))
    p4_interdist= intraclass_distances(np.asarray(p4_samples))
    p5_interdist= intraclass_distances(np.asarray(p5_samples))
    return np.mean([stable_interdist,p1_interdist,p2_interdist,p3_interdist,p4_interdist,p5_interdist])

def total_interdistance(stable_samples,p1_samples,p2_samples,p3_samples,p4_samples,p5_samples):
    stable_p1 = interclass_distances(stable_samples,p1_samples)
    stable_p2 = interclass_distances(stable_samples,p2_samples)
    stable_p3 = interclass_distances(stable_samples,p3_samples)
    stable_p4 = interclass_distances(stable_samples,p4_samples)
    stable_p5 = interclass_distances(stable_samples,p5_samples)
    p1_p2 = interclass_distances(p1_samples, p2_samples)
    p1_p3 = interclass_distances(p1_samples, p3_samples)
    p1_p4 = interclass_distances(p1_samples, p4_samples)
    p1_p5 = interclass_distances(p1_samples, p5_samples)
    p2_p3 = interclass_distances(p2_samples,p3_samples)
    p2_p4 = interclass_distances(p2_samples,p4_samples)
    p2_p5 = interclass_distances(p2_samples,p5_samples)
    p3_p4 = interclass_distances(p3_samples,p4_samples)
    p3_p5 = interclass_distances(p3_samples,p5_samples)
    p4_p5 = interclass_distances(p4_samples,p5_samples)
    return np.mean([stable_p1,stable_p2,stable_p3,stable_p4,stable_p5,p1_p2,p1_p3,p1_p4,p1_p5,p2_p3,p2_p4,p2_p5,p3_p4,p3_p5, p4_p5])

def total_interdistance2(mean_stable,mean_p1,mean_p2,mean_p3,mean_p4,mean_p5):
    stable_p1 = interclass_distances2(mean_stable,mean_p1)
    stable_p2 = interclass_distances2(mean_stable,mean_p2)
    stable_p3 = interclass_distances2(mean_stable,mean_p3)
    stable_p4 = interclass_distances2(mean_stable,mean_p4)
    stable_p5 = interclass_distances2(mean_stable,mean_p5)
    p1_p2 = interclass_distances2(mean_p1, mean_p2)
    p1_p3 = interclass_distances2(mean_p1, mean_p3)
    p1_p4 = interclass_distances2(mean_p1, mean_p4)
    p1_p5 = interclass_distances2(mean_p1, mean_p5)
    p2_p3 = interclass_distances2(mean_p2,mean_p3)
    p2_p4 = interclass_distances2(mean_p2,mean_p4)
    p2_p5 = interclass_distances2(mean_p2,mean_p5)
    p3_p4 = interclass_distances2(mean_p3,mean_p4)
    p3_p5 = interclass_distances2(mean_p3,mean_p5)
    p4_p5 = interclass_distances2(mean_p4,mean_p5)
    return np.mean([stable_p1,stable_p2,stable_p3,stable_p4,stable_p5,p1_p2,p1_p3,p1_p4,p1_p5,p2_p3,p2_p4,p2_p5,p3_p4,p3_p5, p4_p5])

def total_intradistance2(stable_samples,p1_samples,p2_samples,p3_samples,p4_samples,p5_samples,mean_stable,mean_p1,mean_p2,mean_p3,mean_p4,mean_p5):
    stable_interdist = calculateMaxRadius(stable_samples, mean_stable)
    p1_interdist= calculateMaxRadius(p1_samples, mean_p1)
    p2_interdist= calculateMaxRadius(p2_samples,mean_p2)
    p3_interdist= calculateMaxRadius(p3_samples,mean_p3)
    p4_interdist= calculateMaxRadius(p4_samples,mean_p4)
    p5_interdist= calculateMaxRadius(p5_samples,mean_p5)
    return np.mean([stable_interdist,p1_interdist,p2_interdist,p3_interdist,p4_interdist,p5_interdist])
def separate_to_groups(data,labels):
    d_stable=[]
    d_p1=[]
    d_p2=[]
    d_p3=[]
    d_p4=[]
    d_p5=[]
    for i, el in enumerate (data):
        if labels[i]==0:
            d_stable.append(el)
        elif labels[i]==1:
            d_p1.append(el)
        elif labels[i]== 2:
            d_p2.append(el)
        elif labels[i]== 3:
            d_p3.append(el)
        elif labels[i]== 4:
            d_p4.append(el)
        elif labels[i]== 5:
            d_p5.append(el)
    return d_stable,d_p1,d_p2,d_p3,d_p4,d_p5

def results_analysis(d_original, d_100, marge):
    count = 0
    for i in d_100:
        if i < d_original + marge:
            count += 1
    return count/100

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
    file_spread = "spread2.txt"
    file_distances= "distances2.txt"
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
    ##### Standardization
    d_stable = preprocessing.scale_select(d_stable)
    d_p1 = preprocessing.scale_select(d_p1)
    d_p2 = preprocessing.scale_select(d_p2)
    d_p3 = preprocessing.scale_select(d_p3)
    d_p4 = preprocessing.scale_select(d_p4)
    d_p5 = preprocessing.scale_select(d_p5)

    ### Original clusters distances
    stable_samples = select_10samples(d_stable)
    p1_samples = select_10samples(d_p1)
    p2_samples = select_10samples(d_p2)
    p3_samples = select_10samples(d_p3)
    p4_samples = select_10samples(d_p4)
    p5_samples = select_10samples(d_p5)
    ##### Calculate mean vectors for each group
    mean_stable = np.mean(stable_samples, axis=0)
    mean_p1 = np.mean(p1_samples, axis=0)
    mean_p2 = np.mean(p2_samples, axis=0)
    mean_p3 = np.mean(p3_samples, axis=0)
    mean_p4 = np.mean(p4_samples, axis=0)
    mean_p5 = np.mean(p5_samples, axis=0)

    totalclusters_intradist2 =  total_intradistance2(stable_samples,p1_samples,p2_samples,p3_samples,p4_samples,p5_samples,mean_stable,mean_p1,mean_p2,mean_p3,mean_p4,mean_p5)
    print("intra")
    totalclusters_interdist2 = total_interdistance2(mean_stable,mean_p1,mean_p2,mean_p3,mean_p4,mean_p5)
    print("inter")
    d_original2 = totalclusters_intradist2/totalclusters_interdist2
    fd = open(file_distances, "w")
    fd.write("Original distances")
    print("D original for min radius: "+ str(d_original2))
    fd.write("\nTotal original distance for max radius:  " + str(d_original2))

    #### Now we should calculate D for 100 times, when changing the labels of the groups
    data = np.concatenate((d_stable, d_p1, d_p2, d_p3, d_p4, d_p5))
    random.shuffle(data)
    d_total = []
    for i in range(100):
        labels = []
        for x in range(len(data)):
            labels.append(random.randrange(6))
        stable, p1, p2, p3, p4, p5 = separate_to_groups(data, labels)
        stable_samples = select_10samples(stable)
        p1_samples = select_10samples(p1)
        p2_samples = select_10samples(p2)
        p3_samples = select_10samples(p3)
        p4_samples = select_10samples(p4)
        p5_samples = select_10samples(p5)

        ##### Calculate mean vectors for each group
        mean_stable = np.mean(stable_samples, axis=0)
        mean_p1 = np.mean(p1_samples, axis=0)
        mean_p2 = np.mean(p2_samples, axis=0)
        mean_p3 = np.mean(p3_samples, axis=0)
        mean_p4 = np.mean(p4_samples, axis=0)
        mean_p5 = np.mean(p5_samples, axis=0)

        totalclusters_interdist2 = total_interdistance2(mean_stable,mean_p1,mean_p2,mean_p3,mean_p4,mean_p5)
        totalclusters_intradist2 = total_intradistance2(stable_samples, p1_samples, p2_samples, p3_samples, p4_samples,
                                                      p5_samples,mean_stable,mean_p1,mean_p2,mean_p3,mean_p4,mean_p5)
        d_total.append(totalclusters_intradist2 / totalclusters_interdist2 )

    fd.write("\n100 random distances:")
    # print(np.mean(d_total))
    print("D total for max radius: ")
    print(d_total)
    fd.write("\nTotal mean of 100 distance for max radius:  " + str(np.mean(d_total)))

    #comparison of total distances
    prob_0 = results_analysis(d_original2, d_total, 0.0)
    fd.write("\nD original is bigger than in 100 random cases with a marge=0 and probability = :  " + str(prob_0))
    prob_05 = results_analysis(d_original2, d_total, 0.05)
    fd.write("\nD original is bigger than in 100 random cases with a marge=0.05 and probability = :  " + str(prob_05))
    prob_1 = results_analysis(d_original2, d_total, 0.1)
    fd.write("\nD original is bigger than in 100 random cases with a marge=0.1 and  probability = :  " + str(prob_1))
    prob_5 = results_analysis(d_original2, d_total, 0.5)
    fd.write("\nD original is bigger than in 100 random cases with a marge=0.5 and  probability = :  " + str(prob_5))

if __name__ == "__main__":
    main()
