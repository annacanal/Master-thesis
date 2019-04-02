def read(file):
    with open(file, "r") as f:
        files = f.readlines() # Read the whole file at once
    files = [x.strip() for x in files]
    return files

def main():
    mcinames = read("MCInames.txt")
    adnames = read("ADnames.txt")
    cnnames = read("CNnames.txt")
    file = open("mci_bet.txt", "w")
    file2 = open("ad_bet.txt", "w")
    file3 = open("cn_bet.txt", "w")
    for i in mcinames:
        file.write("/usr/local/fsl/bin/bet /media/acanal/Elements/MCI_baseline/"+i+" /media/acanal/Elements/MCI_baseline_processed/"+i+ "_brain  -f 0.5 -g 0\n")
    for i in adnames:
        file2.write("/usr/local/fsl/bin/bet /media/acanal/Elements/AD_baseline/"+i+ " /media/acanal/Elements/AD_baseline_processed/"+i+ "_brain  -f 0.5 -g 0\n")
    for i in cnnames:
        file3.write("/usr/local/fsl/bin/bet /media/acanal/Elements/CN_baseline/"+i+" /media/acanal/Elements/CN_baseline_processed/"+i+ "_brain  -f 0.5 -g 0\n")


if __name__ == "__main__":
    main()
