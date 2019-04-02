import numpy as np
import csv, os
import pandas as pd


class patient:
    kind = 'MCI'  # class variable shared by all instances
    name = ""

    def __init__(self, RID, PTID):
        self.ID = RID  # instance variable unique to each instance
        self.DX = []  # List of DX [MCI, AD]
        self.visit = []  # List of visits ['m06', 'm24']
        self.PTID = PTID  # instance variable unique to each instance
        self.stable = 2  # stable = 0 or stable = 1

    def add_DX(self, DX, visit):
        self.DX.append(DX)
        self.visit.append(visit)

    def stability(self, stable):
        self.stable = stable  # stable= 1 patient remained MCI, stable=0 patient evolved to dementia (AD)

    def add_image_name(self, name):
        self.name = name  # image name in order to load it


def read_ADNIMERGE(data_path,filename):
    df = pd.read_excel(os.path.join(data_path, filename), sheetname='ADNIMERGE')
    DX = pd.Series.tolist(df['DX'])
    viscode = pd.Series.tolist(df['VISCODE'])
    PTID = pd.Series.tolist(df['PTID'])
    RID = pd.Series.tolist(df['RID'])
    return DX,viscode,PTID,RID


def generate_dictionaries(RID,PTID):
    rid = set(RID)  # em quedo amb els valors no repetits, és a dir amb el numero de pacients total
    counts = []
    for i in rid:
        counts.append(RID.count(i))

    # crear diccionari RID i PTID
    dictionary = dict(zip(RID, PTID))
    # crear diccionari rid i quantes vegades surt aquell rid
    dictionary2 = dict(zip(rid, counts))

    return dictionary, dictionary2, counts

def create_patient_list(RID,dictionary,viscode,DX, dictionary2):
    patients = []
    for index, el in enumerate(RID):
        if viscode[index] == 'bl' and DX[index] == "MCI":
            x = patient(el, dictionary[el])
            x.add_DX(DX[index], viscode[index])
            patients.append(x)

    for i, p in enumerate(patients):
        stable = 1
        for c in range(dictionary2[p.ID]):
            index = RID.index(p.ID)
            if DX[index + c] == "Dementia" and stable == 1:
                stable = 0
                patients[i].add_DX("Dementia", viscode[index + c])
                patients[i].stability(stable)
            if c == dictionary2[p.ID] - 1 and stable == 1:
                patients[i].add_DX("MCI", viscode[index + c])
                patients[i].stability(stable)
    return patients

def cluster_patients(patients):
    stable = []  # remained MCI
    progressive_1 = []  # less than 1 year (included)
    progressive_2 = []  # more than 1 year and less than 2 years (included)
    progressive_3 = []  # more than 2 years and less than 3 years (included)
    progressive_4 = []  # more than 3 years and less than 5 years (included)
    progressive_5 = []  # more than 5 years
    for p in patients:
        x = patient(p.ID, p.PTID)
        x.add_DX(p.DX, p.visit)
        if p.stable == 1:
            stable.append(x)
        elif p.stable == 0:
            if p.visit[1] == 'm12' or p.visit[1] == 'm06':
                progressive_1.append(x)
            elif p.visit[1] == 'm18' or p.visit[1] == 'm24':
                progressive_2.append(x)
            elif p.visit[1] == 'm36' or p.visit[1] == 'm30':
                progressive_3.append(x)
            elif p.visit[1] == 'm42' or p.visit[1] == 'm48' or p.visit[1] == 'm54' or p.visit[1] == 'm60':
                progressive_4.append(x)
            else:
                progressive_5.append(x)

    return stable, progressive_1, progressive_2,  progressive_3, progressive_4  , progressive_5


def read(file):
    with open(file, "r") as f:
        files = f.readlines() # Read the whole file at once
    files = [x.strip() for x in files]
    return files

def generate_cluster_folders(mci_file,stable, progressive_1, progressive_2,  progressive_3, progressive_4 , progressive_5):
    mcinames = read(mci_file)
    file = open("mci_stability.txt", "w")
    file0 = open("mci_stable.txt", "w")
    file1 = open("progressive1.txt", "w")
    file2 = open("progressive2.txt", "w")
    file3 = open("progressive3.txt", "w")
    file4 = open("progressive4.txt", "w")
    file5 = open("progressive5.txt", "w")

    for i in stable:
        for j in mcinames:
            if i.PTID in j:
                #file.write("mv  " + j +" /media/acanal/Elements/MCI_baseline_processed/mci_stable\n")
                file.write("mv  " + j +" /media/acanal/Elements/MCI_data/mci_stable\n")
                file0.write(j +"\n")
    for i in progressive_1:
        for j in mcinames:
            if i.PTID in j:
                # file.write("mv  " + j +" /media/acanal/Elements/MCI_baseline_processed/progressive1\n")
                file.write("mv  " + j +" /media/acanal/Elements/MCI_data/progressive1\n")
                file1.write(j +"\n")
    for i in progressive_2:
        for j in mcinames:
            if i.PTID in j:
                # file.write("mv  " + j +" /media/acanal/Elements/MCI_baseline_processed/progressive2\n")
                file.write("mv  " + j + " /media/acanal/Elements/MCI_data/progressive2\n")
                file2.write(j +"\n")
    for i in progressive_3:
        for j in mcinames:
            if i.PTID in j:
                # file.write("mv " + j +" /media/acanal/Elements/MCI_baseline_processed/progressive3\n")
                file.write("mv " + j +" /media/acanal/Elements/MCI_data/progressive3\n")
                file3.write(j +"\n")
    for i in progressive_4:
        for j in mcinames:
            if i.PTID in j:
                # file.write("mv " + j +" /media/acanal/Elements/MCI_baseline_processed/progressive4\n")
                file.write("mv " + j +" /media/acanal/Elements/MCI_data/progressive4\n")
                file4.write(j +"\n")
    for i in progressive_5:
        for j in mcinames:
            if i.PTID in j:
                # file.write("mv " + j + " /media/acanal/Elements/MCI_baseline_processed/progressive5\n")
                file.write("mv " + j + " /media/acanal/Elements/MCI_data/progressive5\n")
                file5.write(j +"\n")


def main():
    data_root = "../Data_Database"
    filename = "ADNIMERGE.xlsm"
    # mci_file = "MCInames_processed.txt"
    mci_file = "mci.txt"

    #### Read ADNIMERGE info
    DX, viscode, PTID, RID = read_ADNIMERGE(data_root,filename)
    dictionary, dictionary2, counts = generate_dictionaries(RID,PTID)
    patients = create_patient_list(RID, dictionary, viscode, DX, dictionary2)
    #### Separate patients regarding their stability or progression
    stable, progressive_1, progressive_2,  progressive_3, progressive_4 , progressive_5 = cluster_patients(patients)

    print(len(stable))
    print(len(progressive_1))
    print(len(progressive_2))
    print(len(progressive_3))
    print(len(progressive_4))
    print(len(progressive_5))

    generate_cluster_folders(mci_file,stable, progressive_1, progressive_2, progressive_3, progressive_4, progressive_5)

if __name__ == "__main__":
    main()
