import numpy as np
import os
import tSNE_visualization
from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import preprocessing
PERPLEXITY = 30
LEARNING_RATE = 600
EXAGGERATION = 80

def evaluate(test_labels, predictions):
    precision, recall, f1score, support = precision_recall_fscore_support(test_labels, predictions)
    print("eval")
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(f1score))
    print('mean precision: {}'.format(np.mean(precision)))
    print('mean recall: {}'.format(np.mean(recall)))
    print('mean fscore: {}'.format(np.mean(f1score)))
    return precision, recall, f1score

def mean_accuracy(test_labels, predictions):
    return accuracy_score(test_labels,predictions)

def main():
    data_path = "../Results/CAE_DBSCAN/images606040"
    data_path1 = "../Results/CAE_DBSCAN/images12012080"
    data_path2 = "../Results/CAE_kmeans/950 images"
    data_path3 = "../Results/CAE_kmeans/images12012080"
    data_path256 = "../Results/Results_train_256_3conv"
    data_path192 = "../Results/Results_train_192_3conv"
    d192 = "../Results/results_192_pdc"
    d256 = "../Results/results_256_pdc"
    modeltype ="CAE"

    #Load labels_test and compressed data
    labels_test256_5 = np.load(os.path.join(d256, "labels_test256_5.npy"))
    labels_test256_10 = np.load(os.path.join(d256, "labels_test256_5.npy"))
    compressed256_conv1_5 = np.load(os.path.join(d256, "compressed256_conv1_5.npy"))
    compressed256_conv1_10 = np.load(os.path.join(d256, "compressed256_conv1_10.npy"))
    compressed256_conv2_5 = np.load(os.path.join(d256, "compressed256_conv2_5.npy"))
    compressed256_conv2_10 = np.load(os.path.join(d256, "compressed256_conv2_10.npy"))
    compressed256_conv3_5 = np.load(os.path.join(d256, "compressed256_conv3_5.npy"))
    compressed256_conv3_10 = np.load(os.path.join(d256, "compressed256_conv3_10.npy"))

    labels_test192_5 = np.load(os.path.join(d192, "labels_test192_5.npy"))
    labels_test192_10 = np.load(os.path.join(d192, "labels_test192_10.npy"))
    compressed192_conv1_5 = np.load(os.path.join(d192, "compressed192_conv1_5.npy"))
    compressed192_conv1_10 = np.load(os.path.join(d192, "compressed192_conv1_10.npy"))
    compressed192_conv2_5 = np.load(os.path.join(d192, "compressed192_conv2_5.npy"))
    compressed192_conv2_10 = np.load(os.path.join(d192, "compressed192_conv2_10.npy"))
    compressed192_conv3_5 = np.load(os.path.join(d192, "compressed192_conv3_5.npy"))
    compressed192_conv3_10 = np.load(os.path.join(d192, "compressed192_conv3_10.npy"))

    ##TSNE REPRESENTATIOM
    print("Images 256, batch_size= 5")
    reduced256_conv1_5, tsne = tSNE_visualization.reduction(compressed256_conv1_5, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced256_conv1_5, labels_test256_5, "Norm Images 256 after conv1, batch_size = 5",d256, type="actual")
    reduced256_conv2_5, tsne = tSNE_visualization.reduction(compressed256_conv2_5, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced256_conv2_5, labels_test256_5, "Norm Images 256 after conv2, batch_size = 5",d256, type="actual")
    reduced256_conv3_5, tsne = tSNE_visualization.reduction(compressed256_conv3_5, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced256_conv3_5, labels_test256_5, "Norm Images 256 after conv3, batch_size = 5",d256, type="actual")
    print("Images 256, batch_size= 10")
    reduced256_conv1_10, tsne = tSNE_visualization.reduction(compressed256_conv1_10, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced256_conv1_10, labels_test256_10, "Norm Images 256 after conv1, batch_size = 10",d256, type="actual")
    reduced256_conv2_10, tsne = tSNE_visualization.reduction(compressed256_conv2_10, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced256_conv2_10, labels_test256_10, "Norm Images 256 after conv2, batch_size = 10",d256, type="actual")
    reduced256_conv3_10, tsne = tSNE_visualization.reduction(compressed256_conv3_10, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced256_conv3_10, labels_test256_10, "Norm Images 256 after conv3, batch_size = 10",d256, type="actual")
    print("Images 192, batch_size= 5")
    reduced192_conv1_5, tsne = tSNE_visualization.reduction(compressed192_conv1_5, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced192_conv1_5, labels_test192_5, "Norm Images 192 after conv1, batch_size = 5",d192, type="actual")
    reduced192_conv2_5, tsne = tSNE_visualization.reduction(compressed192_conv2_5, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced192_conv2_5, labels_test192_5, "Norm Images 192 after conv2, batch_size = 5",d192, type="actual")
    reduced192_conv3_5, tsne = tSNE_visualization.reduction(compressed192_conv3_5, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced192_conv3_5, labels_test192_5, "Norm Images 192 after conv3, batch_size = 5", d192, type="actual")
    print("Images 192, batch_size= 10")
    reduced192_conv1_10, tsne = tSNE_visualization.reduction(compressed192_conv1_10, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced192_conv1_10, labels_test192_10, "Norm Images 192 after conv1, batch_size = 10",d192, type="actual")
    reduced192_conv2_10, tsne = tSNE_visualization.reduction(compressed192_conv2_10, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced192_conv2_10, labels_test192_10, "Norm Images 192 after conv2, batch_size = 10",d192, type="actual")
    reduced192_conv3_10, tsne = tSNE_visualization.reduction(compressed192_conv3_10, PERPLEXITY, l_r=LEARNING_RATE, dim=2, ex=EXAGGERATION)
    tSNE_visualization.plot_clusters(reduced192_conv3_10, labels_test192_10, "Norm Images 192 after conv3, batch_size = 10",d192, type="actual")


if __name__ == "__main__":
    main()
