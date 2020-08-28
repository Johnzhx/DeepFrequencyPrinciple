import tensorflow as tf
import keras
import os
import numpy as np
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, ZeroPadding2D, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.utils.np_utils import *
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.backend as K
from matplotlib.pyplot import imshow
import scipy.misc
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import pydot
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.preprocessing import image
import pickle
import sys

np.random.seed(1)
allset = np.random.permutation(50000)
allset_train = allset[10000:50000]
allset_validation = allset[0:10000]
print(allset)

R={}

tf.compat.v1.set_random_seed(40)

def mkdir(fn): 
    if not os.path.isdir(fn):
        os.mkdir(fn)

FolderName = r'loss_and_acc_-4'+'/'
R['FolderName'] = FolderName
mkdir(FolderName)

def savefile(): 
    with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(R, f, protocol=4)
    text_file = open("%s/Output.txt"%(FolderName), "w")
    for para in R:
        if np.size(R[para])>20:
            continue
        text_file.write('%s: %s\n'%(para,R[para]))
    
    for para in sys.argv: 
        text_file.write('%s  '%(para))
    text_file.close()

os.environ["CUDA_VISIBLE_DEVICES"]='3' 

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images0, test_images0 = train_images / 255.0, test_images / 255.0
train_labels0=keras.utils.to_categorical(train_labels, num_classes = 10)
test_labels0=keras.utils.to_categorical(test_labels, num_classes = 10)

train_images=train_images0[allset_train,:,:,:]
train_labels=train_labels0[allset_train,:]
validation_images = train_images0[allset_validation,:,:,:]
validation_labels = train_labels0[allset_validation,:]
test_images=test_images0[0:10000,:,:,:]
test_labels=test_labels0[0:10000,:]

def Resnet18_block(X, filters, s ,stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2 = filters
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    if s != 1:
        X_shortcut = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), padding = 'same', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=3))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X_shortcut)
        X_shortcut = Activation('relu')(X_shortcut)       
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (3, 3), strides = (s,s), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=4))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=5))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def ResNet18(input_shape = (32,32,3),classes=10):
    X_input = Input(input_shape)
    #X1 = ZeroPadding2D((3,3))(X_input)
    #stage 1 
    X1 = Conv2D(64,(3,3),strides = (1,1),name = 'conv1',padding='same', kernel_initializer = glorot_uniform(seed=2))(X_input)
    X3 = BatchNormalization(axis = 3, name = 'bn_conv1')(X1)
    X2 = Activation('relu')(X3)
    X32 = Dropout(0.1)(X2)

    X11 = AveragePooling2D(pool_size=(4,4))(X32)
    X12 = Dropout(0.1)(X11)
    
    X23 = Flatten()(X12)
    X24 = Dense(1024,activation='relu', kernel_initializer = glorot_uniform(seed=0),name='dense1')(X23)
    X25 = Dense(1024,activation='relu', kernel_initializer = glorot_uniform(seed=0),name='dense2')(X24)
    X26 = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X25)

    model = Model(inputs = X_input, outputs = X26, name='ResNet18')

    return model

model = ResNet18(input_shape = (32,32,3),classes=10)
model.summary()

adam2 = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam2'
        )
adam4 = tf.keras.optimizers.Adam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam4'
        )
adam5 = tf.keras.optimizers.Adam(
        learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam5'
        )

model.compile(optimizer=adam2,
              loss= tf.keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=2)
history1 = model.fit(train_images, train_labels, epochs=40, 
                     batch_size=256,
                     validation_data=(validation_images, validation_labels),
                     shuffle=True)
hist_val_loss=history1.history['val_loss']
hist_loss=history1.history['loss']
hist_acc=history1.history['accuracy']
hist_val_acc=history1.history['val_accuracy']

model.compile(optimizer=adam4,
              loss= tf.keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
history2 = model.fit(train_images, train_labels, epochs=20, 
                     batch_size=256,
                     validation_data=(validation_images,validation_labels),
                     shuffle=True)
hist_val_loss=hist_val_loss+history2.history['val_loss']
hist_loss=hist_loss+history2.history['loss']
hist_acc=hist_acc+history2.history['accuracy']
hist_val_acc=hist_val_acc+history2.history['val_accuracy']

model.compile(optimizer=adam5,
              loss= tf.keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])
history3 = model.fit(train_images, train_labels, epochs=20, 
                    batch_size=256,
                    validation_data=(validation_images,validation_labels),
                    shuffle=True)
hist_val_loss=hist_val_loss+history3.history['val_loss']
hist_loss=hist_loss+history3.history['loss']
hist_acc=hist_acc+history3.history['accuracy']
hist_val_acc=hist_val_acc+history3.history['val_accuracy']

(hist_test_loss,hist_test_acc) = model.evaluate(test_images , test_labels, batch_size=128)

R['hist_val_loss']=hist_val_loss
R['hist_loss']=hist_loss
R['hist_acc']=hist_acc
R['hist_val_acc']=hist_val_acc
R['hist_test_loss']=hist_test_loss
R['hist_test_acc']=hist_test_acc

savefile()

model.save('my_model_Resnet18-4_80.h5') 
del model  
model = load_model('my_model_Resnet18-4_80.h5')
