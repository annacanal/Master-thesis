import numpy as np
import os


def read_images(data_path1, data_path2, data_path3, filename1,filename2,filename3,n_mci, n_ad, n_cn):
    images = []
    names= []
    labels = []
    ######## MCI ########
    with open(os.path.join(data_path1, filename1), "r") as f:
        files1 = f.readlines() # Read the whole file at once
    files1 = [x.strip() for x in files1]
    #for i in files1:
    for i in range(n_mci):
        names.append(files1[i])
        im = np.load(os.path.join(data_path1, files1[i]))
        images.append(im['arr_0'])
        labels.append(0) # 0 = MCI)
    ######## AD ########
    with open(os.path.join(data_path2,filename2), "r") as f:
        files2 = f.readlines() # Read the whole file at once
    files2 = [x.strip() for x in files2]
    for i in range(n_ad):
        names.append(files2[i])
        im = np.load(os.path.join(data_path2, files2[i]))
        images.append(im['arr_0'])
        labels.append(1) # 1=AD
    ######## CN ########
    with open(os.path.join(data_path3,filename3), "r") as f:
        files3 = f.readlines() # Read the whole file at once
    files3 = [x.strip() for x in files3]
    for i in range(n_cn):
        names.append(files3[i])
        im = np.load(os.path.join(data_path3, files3[i]))
        images.append(im['arr_0'])
        labels.append(2)  #2 = CN
    return images, labels, names

