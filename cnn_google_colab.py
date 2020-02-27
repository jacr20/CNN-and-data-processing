#import comet_ml
import numpy as np
import glob
import os.path as path
from scipy import misc
import os
import pandas as pd
import random
import imageio
import tensorflow
from tensorflow import keras

from PIL import Image
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,Activation, AveragePooling2D,PReLU, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.backend import cast, greater, clip, floatx,epsilon
#from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K
#from comet_ml import Experiment
from sklearn.model_selection import StratifiedKFold

#experiment = Experiment("uiydhIvcMtRiUnJEBRltGfvn2")

np.random.seed(0)
random.seed(0)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

IMAGE_PATH = '/content/20000_75x75'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))
# Load the images
images = [imageio.imread(path) for path in file_paths]
images = np.asarray(images)
# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
# Scale
images = images / 255

# Read the labels from the filenames
n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    if filename[0] == 'W':
        labels[i] = 1
    else:
        labels[i] = 0

#WIMPS = 1 = True
#Background = 0 = FALSE
# Split the images and the labels
Train = images
Y_train = labels





def create_model(conv_layers, N_c_neurons, KERNEL, stride_1, stride_2, alpha_1, alpha_2, p_size, fc_layers, N_fc_neurons, alpha_3, alpha_4, alpha_5):
  model = Sequential()
  for i in range(conv_layers):
    model.add(Conv2D(N_c_neurons, KERNEL, strides = stride_1, kernel_regularizer=l2(alpha_1)))
    model.add(keras.layers.LeakyReLU(alpha = alpha_2))
    model.add(MaxPooling2D(pool_size = p_size, strides = stride_2))
    
  model.add(Dropout(0.25))
    
  model.add(Flatten)

  for i in range(fc_layers):
    model.add(Dense(N_fc_neurons, bias_regularizer=l2(alpha_3), kernel_regularizer=l2(alpha_4)))
    model.add(LeakyReLU(alpha=alpha_5))
  
  model.add(Dropout(0.25))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[recall_m])
  return model

model = KerasClassifier(build_fn=create_model, verbose=1, conv_layers = 3, N_c_neurons = 20, KERNEL = (3,3), stride_1 = (1,1), stride_2 = (1,1), alpha_1 = 0.005, alpha_2 = 0.05, p_size = (2,2), fc_layers = 1, N_fc_neurons = 20, alpha_3 = 0.001, alpha_4 = 0.001, alpha_5 = 0.05)

params_to_test_gen = {'epochs':[10, 50 , 100], 'batch_size': [10, 50, 100, 150]}

#parmas_to_test_archs = {'conv_layers':[1,2,3], 'fc_layers':[1,2,3],'N_c_neurons':[10, 20, 30], 'N_fc_neurons':[10, 20, 30]}

#params_to_test_grids = {'KERNEL':[(2,2), (3,3), (4,4)], 'p_size':[(2,2), (3,3), (4,4)] , 'stride_1':[(1,1), (2,2), (3,3)], 'stride_2':[(1,1), (2,2), (3,3)]}

#params_to_test_regs = 'alpha_1':[0.001, 0.005, 0.01], 'alpha_2':[0.005, 0.05, 0.1], 'alpha_3':[0.0005, 0.001, 0.05], 'alpha_4':[0.0005, 0.001, 0.05], 'alpha_5':[0.005, 0.05, 0.1]}

'''Run this gird search for each seperate grid param, once you have the best params fix them in the model above and do the next search.'''
gsearch_0 = GridSearchCV(estimator = model, param_grid = params_to_test_gen, scoring='recall', n_jobs= None, iid=False, cv=StratifiedKFold(n_splits=3, shuffle = True, random_state = 0), verbose = 5)
gsearch_0.fit(Train, Y_train)
print(gsearch_0.best_score_)
print(gsearch_0.best_params_)

#gsearch_1 = GridSearchCV(estimator = model, param_grid = params_to_test_arch, scoring='recall', n_jobs=-1, iid=False, cv=StratifiedKFold(n_splits=3, shuffle = True, random_state = 0), verbose = 5)
#gsearch_1.fit(Train, Y_train)
#print(gsearch_1.best_score_)
#print(gsearch_1.best_params_)

#gsearch_2 = GridSearchCV(estimator = model, param_grid = params_to_test_grids, scoring='recall', n_jobs=-1, iid=False, cv=3, verbose = 5)
#gsearch_2.fit(Train, Y_train)
#print(gsearch_2.best_score_)
#print(gsearch_2.best_params_)

#gsearch_3 = GridSearchCV(estimator = model, param_grid = params_to_test_regs, scoring='recall', n_jobs=-1, iid=False, cv=3, verbose = 5)
#gsearch_3.fit(Train, Y_train)
#print(gsearch_3.best_score_)
#print(gsearch_3.best_params_)

#experiment.end()
