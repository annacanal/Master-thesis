import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import CAE2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import get_compressed_data
import preprocessing
from time import time

################# SLICES 2D, option 1: [:,:, X] ###################
IM_X = 80
IM_Y = 80

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
    learning_rate = 0.1


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)  # patience is the number of epochs with no improvement after which training will be stopped
    check = ModelCheckpoint("/tmp/cae80_slices1.hdf5", monitor= 'val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    tensorboard = TensorBoard(log_dir="logs80slice1/{}".format(time()), histogram_freq=1, batch_size=30, write_graph=True, write_grads=True,
                              write_images=True, embeddings_freq=1)
    callbacks = [early_stopping, reduce_lr, tensorboard, check]


    #Load Data
    images,labels, names= get_compressed_data.read_images(data_pathMCI,data_pathAD,data_pathCN, fileMCI, fileAD, fileCN, n_mci, n_ad, n_cn)
    # Shuffle data and Split into train and test
    images_train, images_test, labels_train, labels_test, names_train, names_test= train_test_split(images,labels,names,test_size=0.50, random_state=42)
    # Cut images in [80,80,80], save them in arrays train and test
    train = np.zeros((len(images_train), IM_X, IM_Y))
    test = np.zeros((len(images_test), IM_X, IM_Y))
    #train images
    for i,element in enumerate(images_train):
        train[i] = element[5:85,15:95,45]
    #test images
    for i,element in enumerate(images_test):
        test[i] = element[5:85,15:95,45]
    train = train.astype('float32')/255
    test = test.astype('float32')/255

    print(train.shape)
    # Standardization
    reshaped_train = train.reshape(len(images_train), IM_X * IM_Y)
    train_normalized = preprocessing.scale_select(reshaped_train)
    reshaped_test = test.reshape(len(images_test), IM_X * IM_Y)
    test_normalized = preprocessing.scale_select(reshaped_test)
    train = train_normalized.reshape(len(images_train), IM_X , IM_Y)
    test = test_normalized.reshape(len(images_test), IM_X , IM_Y)
    np.save("labels_test", labels_test)

    for n_batch in batch:
        train = train.reshape(train.shape[0], IM_X, IM_Y, 1)  # transform 3D matrix to 4D, 4 dimension is number of channels
        test = test.reshape(test.shape[0], IM_X, IM_Y, 1)  # channels=1 gray scale, channel=3 RGB
        for lr in learning_rate:
            model = CAE2D.create_model(train[0].shape)
            model= CAE2D.compile(model, lrate = lr)
            model= CAE2D.train(model, train, test, callbacks, batch_size= n_batch, epochs=epochs)
            # Take compressed layer for clustering
            get_compressed_layer_output512 = K.function([model.layers[0].input], [model.layers[10].output])
            compressed512 = get_compressed_layer_output512([test])[0]
            compressed512 = compressed512.reshape(len(labels_test), 512)  # numero 8 fa referencia quants filtres la layer
            get_compressed_layer_output128 = K.function([model.layers[0].input], [model.layers[11].output])
            compressed128 = get_compressed_layer_output128([test])[0]
            compressed128 = compressed128.reshape(len(images_test), 128)  # numero 8 fa referencia quants filtres la layer
            np.save("cae80slices1_512nodes_"+str(n_batch)+"_lr"+str(lr), compressed512)
            np.save("cae80slices1_128nodes_" + str(n_batch) + "_lr" + str(lr), compressed128)


if __name__ == "__main__":
    main()
