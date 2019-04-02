from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from keras import optimizers
from keras import losses

def create_model(input_img_size):
    model = Sequential()
    model = encoder(model,input_img_size )
    model = decoder(model)
    return model

def encoder(model, input_img_size ):
    model.add(Conv3D(8, (3, 3, 3), padding='same', input_shape= input_img_size)) # apply 8 filters sized of (3x3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))  # strides=(2,2,2)
    # 2nd conv
    model.add(Conv3D(8, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    # 3rd conv
    model.add(Conv3D(8, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    # FC
    model.add(Flatten())
    model.add(Dense(5500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    return model

def decoder(model ):
    model.add(Dense(5500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8000, kernel_initializer='normal', activation='relu'))
    model.add(Reshape((10, 10, 10, 8)))
    # 4th conv
    model.add(Conv3D(8, (3, 3, 3), padding='same'))#, input_shape=input_img_size))
    model.add(Activation('relu'))
    model.add(UpSampling3D((2, 2, 2)))
    # 5th conv
    model.add(Conv3D(8, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling3D((2, 2, 2)))
    # 6th convolution layer
    model.add(Conv3D(8, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling3D((2, 2, 2)))
    # Decoded
    model.add(Conv3D(1, (3, 3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    return model

def compile(model, lrate = 1.0):
    opt =  optimizers.Adadelta(lr=lrate, rho=0.95, decay=0.0)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy)
    return model

def train(model, train, test, callbacks, batch_size = 5, epochs = 3):
    model.fit(train, train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,validation_data=(test, test))
    return model


