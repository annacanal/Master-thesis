import numpy as np
import nibabel
from nilearn.datasets import load_mni152_template
from nilearn import image
import os


data_path1 = "../ADNI/MCI_baseline_processed/"
data_path2 = "../ADNI/AD_baseline_processed/"
data_path3 = "../ADNI/CN_baseline_processed/"

def read(template):
    images = []
    names= []
    labels = []
    with open(os.path.join(data_path1, "MCInames_processed.txt"), "r") as f:
        files1 = f.readlines() # Read the whole file at once
    files1 = [x.strip() for x in files1]
    with open(os.path.join(data_path2,"ADnames_processed.txt"), "r") as f:
        files2 = f.readlines() # Read the whole file at once
    files2 = [x.strip() for x in files2]
    with open(os.path.join(data_path3,"CNnames_processed.txt"), "r") as f:
        files3 = f.readlines() # Read the whole file at once
    files3 = [x.strip() for x in files3]

    for i in files1:
        names.append(i)
        img = nibabel.load(os.path.join(data_path1, i ))
        images.append(image.resample_to_img(img, template))
        labels.append("MCI")
    for i in files2:
        names.append(i)
        img = nibabel.load(os.path.join(data_path2, i))
        images.append(image.resample_to_img(img, template))
        labels.append("AD")
    for i in files3:
        names.append(i)
        img = nibabel.load(os.path.join(data_path3, i))
        images.append(image.resample_to_img(img, template))
        labels.append("CN")
    return images,labels, names

def main():
    # Template MNI152
    template = load_mni152_template()
    # Read images, labels and names
    files, labels, names = read(template)
    # Save labels and names in the compressed npz format
    np.savez_compressed("labels_baseline_c", labels=labels)
    np.savez_compressed("names_baseline_c", names=names)
    # Convert list of nifti images to a numpy array and save to npz
    for i,element in enumerate(files):
        np.savez_compressed(names[i], element.get_fdata())


if __name__ == "__main__":
    main()
