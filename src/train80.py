import numpy as np
import os
from keras import backend as K
from sklearn.model_selection import train_test_split
import CAE
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,TensorBoard
import get_compressed_data
import preprocessing
import matplotlib.pyplot as plt
from time import time

IM_X = 80
IM_Y = 80
IM_Z = 80


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

    batch = 20
    epochs = 150
    learning_rate = 0.1 #[0.001, 0.1, 1.0]

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                   verbose=1)  # patience is the number of epochs with no improvement after which training will be stopped
    check = ModelCheckpoint("/tmp/cae80.hdf5", monitor='val_loss', verbose=0, save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)
    tensorboard = TensorBoard(log_dir="logs80/{}".format(time()), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                              write_images=False)
    callbacks = [early_stopping, reduce_lr, tensorboard, check]


    #Load Data
    images,labels, names= get_compressed_data.read_images(data_pathMCI,data_pathAD,data_pathCN, fileMCI, fileAD, fileCN, n_mci, n_ad, n_cn)
    # Shuffle data and Split into train and test
    images_train, images_test, labels_train, labels_test, names_train, names_test= train_test_split(images,labels,names,test_size=0.30, random_state=42)
    # Cut images in [80,80,80], save them in arrays train and test
    train = np.zeros((len(images_train), IM_X, IM_Y, IM_Z))
    test = np.zeros((len(images_test), IM_X, IM_Y, IM_Z))
    #train images
    for i,element in enumerate(images_train):
        train[i] = element[5:85,15:95,5:85]
    #test images
    for i,element in enumerate(images_test):
        test[i] = element[5:85,15:95,5:85]

    # Standardization
    reshaped_train = train.reshape(len(images_train), IM_X * IM_Y * IM_Z)
    train_normalized = preprocessing.scale_select(reshaped_train)
    reshaped_test = test.reshape(len(images_test), IM_X * IM_Y * IM_Z)
    test_normalized = preprocessing.scale_select(reshaped_test)
    train = train_normalized.reshape(len(images_train), 80 , 80 , 80)
    test = test_normalized.reshape(len(images_test), 80 , 80 , 80)
    np.save("labels_test", labels_test)

    for n_batch in batch:
        train = train.reshape(train.shape[0], IM_X, IM_Y, IM_Z, 1)  # transform 3D matrix to 4D, 4 dimension is number of channels
        test = test.reshape(test.shape[0], IM_X, IM_Y, IM_Z, 1)  # channels=1 gray scale, channel=3 RGB
        for lr in learning_rate:
            model = CAE.create_model(train[0].shape)
            model= CAE.compile(model, lrate = lr)
            model= CAE.train(model, train, test, callbacks, batch_size= n_batch, epochs=epochs)
            # Take compressed layer for clustering
            get_compressed_layer_output5500 = K.function([model.layers[0].input], [model.layers[10].output])
            compressed5500 = get_compressed_layer_output5500([test])[0]
            compressed5500 = compressed5500.reshape(len(labels_test), 5500)  # numero 8 fa referencia quants filtres la layer
            get_compressed_layer_output128 = K.function([model.layers[0].input], [model.layers[11].output])
            compressed128 = get_compressed_layer_output128([test])[0]
            compressed128 = compressed128.reshape(len(images_test), 128)  # numero 8 fa referencia quants filtres la layer
            np.save("cae80_5500nodes_"+str(n_batch)+"_lr"+str(lr), compressed5500)
            np.save("cae80_128nodes_" + str(n_batch) + "_lr" + str(lr), compressed128)


if __name__ == "__main__":
    main()
