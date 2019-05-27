import glob
import numpy as np
import os.path as path
from scipy import misc
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pylab as plab
import matplotlib.mlab as mlab

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,Activation, AveragePooling2D,PReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.backend import cast, greater, clip, floatx,epsilon

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_recall_curve, auc,recall_score
from datetime import datetime
from matplotlib import cm
from inspect import signature
import pandas as pd

'''Pre-amble for TB '''
now = datetime.utcnow().strftime("%Y/%m/%d__%H:%M:%S")
NAME ='ER-vs-WIMP-CNN_HP-only_LRelu_0.5_20_Adam_0.0001_bs_20_{}'.format(now)

'''DATA PREPROCESSING'''


IMAGE_PATH = '/home/joel/Documents/XENON-ML/Figures/ScaledSIM1_CNN'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))


# Load the images
images = [misc.imread(path) for path in file_paths]
images = np.asarray(images)

# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
#print(image_size)

# Scale
images = images / 255

# Read the labels from the filenames
n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    if filename[0] == 'W':                          #Every file that begins with W is assigned a 1
        labels[i] = 1
    else:
        labels[i] = 0

#WIMPS = 1 = True
#Background = 0 = FALSE

# Split into test and training sets
TRAIN_TEST_SPLIT = 0.9

# Split at the given index
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

# Split the images and the labels
x_train1 = images[train_indices, :, :, :]
y_train1 = labels[train_indices]
x_test = images[test_indices, :, :, :]
y_test = labels[test_indices]

#Split the Images further to obtain a validation set
x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size=0.20, random_state=1)

#Ratio:60,20,20




'''DATA VISUALISATION
def visualize_data(positive_images, negative_images):
    # INPUTS
    # positive_images - Images where the label = 1 (True)
    # negative_images - Images where the label = 0 (False)

    figure = plt.figure()
    count = 0
    for i in range(positive_images.shape[0]):
        count += 1
        figure.add_subplot(2, positive_images.shape[0], count)
        plt.imshow(positive_images[i, :, :])
        plt.axis('off')
        plt.title("1")

    count = 0
    for i in range(negative_images.shape[0]):
        count += 1
        figure.add_subplot(1, negative_images.shape[0], count)
        plt.imshow(negative_images[i, :, :])
        plt.axis('off')
        plt.title("0")
    #plt.show()
       
# Number of positive and negative examples to show
N_TO_VISUALIZE = 0

# Select the first N positive examples
positive_example_indices = (y_train == 1)
positive_examples = x_train[positive_example_indices, :, :]
positive_examples = positive_examples[0:N_TO_VISUALIZE, :, :]

# Select the first N negative examples
negative_example_indices = (y_train == 0)
negative_examples = x_train[negative_example_indices, :, :]
negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]

# Call the visualization function
visualize_data(positive_examples, negative_examples)
'''
'''Presicion and recall'''
'''
def precision(y_true, y_predictions,threshold=0.5):
    """Precision metric.
    Computes the precision over the whole batch using threshold_value.
    """
    y_true1 = tf.convert_to_tensor(y_true.astype(np.float32))
    y_predictions1 = tf.convert_to_tensor(y_predictions.astype(np.float32))
    threshold_value = threshold
    # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
    y_prediction1 = cast(greater(clip(y_predictions1, 0, 1), threshold_value), floatx())
    # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
    true_positives1 = tf.keras.backend.round(tf.keras.backend.sum(clip(y_true1 * y_prediction1, 0, 1)))
    # count the predicted positives
    predicted_positives = tf.keras.backend.sum(y_prediction1)
    # Get the precision ratio
    precision_ratio = true_positives1 / (predicted_positives + epsilon())
    return precision_ratio


def recall(y_true, y_predictions,threshold = 0.5):
    """Recall metric.
    Computes the recall over the whole batch using threshold_value.
    """
    y_true2 = tf.convert_to_tensor(y_true.astype(np.float32))
    y_predictions2 = tf.convert_to_tensor(y_predictions.astype(np.float32))
    threshold_value = threshold
    # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
    y_prediction2 = cast(greater(clip(y_predictions2, 0, 1), threshold_value), floatx())
    # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
    true_positives2 = tf.keras.backend.round(tf.keras.backend.sum(clip(y_true2 * y_prediction2, 0, 1)))
    # Compute the number of positive targets.
    possible_positives = tf.keras.backend.sum(clip(y_true2, 0, 1))
    recall_ratio = true_positives2 / (possible_positives + epsilon())
    return recall_ratio
'''
'''CNN Archietecture'''

N_LAYERS = 2
def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 50
    KERNEL = (2, 2)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
        else:
            model.add(Conv2D(nuerons[1], KERNEL,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)))
        
        #model.add(BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.5))

    # Add max pooling layer
    model.add(AveragePooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(MAX_NEURONS, (2,2), input_shape=shape,kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(MIN_NEURONS,bias_regularizer=regularizers.l2(0.01),kernel_regularizer=regularizers.l2(0.01)))
    #model.add(BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(Dropout(0.5))

    # Add output layer
    model.add(Dense(1))
    #model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)

# Training hyperparamters
EPOCHS = 50
BATCH_SIZE = 100

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = ''
log_dir = "/home/joel/Documents/XENON-ML/TB"#.format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir='TB_logs/{}'.format(NAME), write_graph=True, write_images=True, write_grads=True,histogram_freq=1)


# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1 , validation_data=(x_val, y_val))


# Make a prediction on the test set
test_predictions = model.predict(x_test)
print (test_predictions)
#test_predictions = np.round(test_predictions)

# Report the accuracy
accuracy = accuracy_score(y_test, np.round(test_predictions))
print("Accuracy: " + str(accuracy))
#f1 = f1_score(y_test, np.round(test_predictions))
#print("F1 score: " + str(f1))
average_precision = average_precision_score(y_test, test_predictions)
print("Average precision: " + str(average_precision))


precision, recall, thresholds = precision_recall_curve(y_test, test_predictions)
auc = auc(recall, precision)
recall1 = recall_score(y_test, np.round(test_predictions))
print("recall: " + str(recall1))
print('AUC:' +str(auc))


#Plot PR curve
plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.ylim([0.0, 1.05)
#plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(auc))
plt.legend(loc="lower left")
plt.show()

'''
#Example PR curve plot
for i in range(0,10):
    I = 0.1*i
    auc = auc(recall(y_test,test_predictions,I), precision(y_test,test_predictions,I))
    #AUC=%.2f
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    plt.plot(recall(y_test,test_predictions,I), precision(y_test,test_predictions,I))
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example:threshold=%' %(auc,I))
    plt.show()

#precision = precision_score(y_test, test_predictions)
#print("precision: " + str(precision))
#recall = recall_score(y_test, test_predictions)
#print("recall: " + str(recall))

'''

'''
#IMAGES THAT WERE INCORRECT
def visualize_incorrect_labels(x_data, y_real, y_predicted):
    # INPUTS
    # x_data      - images
    # y_data      - ground truth labels
    # y_predicted - predicted label
    count = 0
    figure = plt.figure()
    incorrect_label_indices = (y_real != y_predicted)
    y_real = y_real[incorrect_label_indices]
    y_predicted = y_predicted[incorrect_label_indices]
    x_data = x_data[incorrect_label_indices, :, :, :]

    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

    for i in range(x_data.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        plt.imshow(x_data[i, :, :, :])
        plt.axis('off')
        plt.title("Predicted: " + str(int(np.round(y_predicted[i]))) + ", Real: " + str(int(y_real[i])), fontsize=10)

    plt.show()

visualize_incorrect_labels(x_test, y_test, np.asarray(test_predictions).ravel())
'''

#Report Confusion Matrix
y_actu = pd.Series(y_test.ravel(), name='Actual')
y_pred = pd.Series(np.round(test_predictions.ravel()), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap='YlGn'):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0,len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    for i in range(len(df_confusion.index)):
        for j in range(len(df_confusion.columns)):
            plt.text(j,i,str(df_confusion.iloc[i,j]))
    plt.show()
    plt.show()

plot_confusion_matrix(df_confusion)

