#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import zipfile
import tensorflow as tf
from zipfile import ZipFile
import glob
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#from zipfile import ZipFile


# In[2]:


os.chdir(r"E:\CSE438\Project\Processed\Combine Cancer non cancer\OSCC")
X = []
y = []
for i in tqdm(os.listdir()):
      img = cv2.imread(i)
      img = cv2.resize(img,(224,224))
      X.append(img)
      y.append((i[0:1]))
      print(i[0:1])
os.chdir(r"E:\CSE438\Project\Processed\Combine Cancer non cancer\Normal")
for i in tqdm(os.listdir()):
      img = cv2.imread(i)
      img = cv2.resize(img,(224,224))
      X.append(img)
for i in range(1,2495):
    y.append('N')
print(y)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X[i], cmap="gray")
    #plt.imshow(X[i])
    plt.axis('off')
plt.show()


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print ("Shape of an image in X_train: ", X_train[0].shape)
print ("Shape of an image in X_test: ", X_test[0].shape)


# In[7]:


from sklearn import preprocessing  # Import the preprocessing module


# In[8]:


le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test)


# In[9]:


print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)


# In[10]:


from keras.applications import vgg16


img_rows, img_cols = 224, 224


vgg = vgg16.VGG16(weights = 'imagenet',
                 include_top = False,
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default
for layer in vgg.layers:
    layer.trainable = False

# Let's print our layers
for (i,layer) in enumerate(vgg.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[11]:


def lw(bottom_model, num_classes):
    """creates the top or head of the model that will be
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


# In[12]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.models import Model


num_classes = 2

FC_Head = lw(vgg, num_classes)

model = Model(inputs = vgg.input, outputs = FC_Head)

print(model.summary())


# In[13]:


from tensorflow.keras.models import Model
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[16]:


history = model.fit(X_train,y_train,
                    epochs= 15,
                    validation_data=(X_test,y_test),
                    verbose = 1,
                    initial_epoch=0)


# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[18]:


from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
print('Accuracy = %.2f' % accuracy)

# Generate confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# Set a cool color palette
cool_palette = sns.color_palette("autumn", as_cmap=True)

# Plot the confusion matrix with adjusted font size and color palette
plt.figure(figsize=(5,4))
sns.set(font_scale=1.2)  # Adjust font scale
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap=cool_palette,
            xticklabels=['OSCC', 'Normal'],
            yticklabels=['OSCC', 'Normal'])
plt.xlabel('Predicted', fontsize=14)  # Adjust label font size
plt.ylabel('True', fontsize=14)  # Adjust label font size
plt.title('Confusion Matrix (Test)', fontsize=16)  # Adjust title font size
plt.show()


# In[19]:


# Get predictions on the training set
y_train_pred = model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_train_true = np.argmax(y_train, axis=1)

accuracy_train = accuracy_score(y_train_true, y_train_pred_classes)
print('Train Accuracy = %.2f' % accuracy_train)

# Generate confusion matrix for training set
confusion_mtx_train = confusion_matrix(y_train_true, y_train_pred_classes)

# Set a cool color palette
cool_palette = sns.color_palette("autumn", as_cmap=True)

# Plot the confusion matrix with adjusted font size and color palette
plt.figure(figsize=(5,4))
sns.set(font_scale=1.2)  # Adjust font scale
sns.heatmap(confusion_mtx_train, annot=True, fmt='d', cmap=cool_palette,
            xticklabels=['OSCC', 'Normal'],
            yticklabels=['OSCC', 'Normal'])
plt.xlabel('Predicted', fontsize=14)  # Adjust label font size
plt.ylabel('True', fontsize=14)  # Adjust label font size
plt.title('Confusion Matrix (Train)', fontsize=16)  # Adjust title font size
plt.show()


# In[20]:


from sklearn import metrics  # Add this line to import the metrics module

# Calculate evaluation metrics
accuracy = np.round(metrics.accuracy_score(y_true, y_pred_classes), 4)
precision = np.round(metrics.precision_score(y_true, y_pred_classes, average='weighted'), 4)
recall = np.round(metrics.recall_score(y_true, y_pred_classes, average='weighted'), 4)
f1_score = np.round(metrics.f1_score(y_true, y_pred_classes, average='weighted'), 4)
roc_auc = np.round(metrics.roc_auc_score(y_test, y_pred, multi_class='ovo', average='weighted'), 4)
cohen_kappa = np.round(metrics.cohen_kappa_score(y_true, y_pred_classes), 4)

# Print the evaluation metrics
print('Accuracy score is :', accuracy)
print('Precision score is :', precision)
print('Recall score is :', recall)
print('F1 Score is :', f1_score)
print('ROC AUC Score is :', roc_auc)
print('Cohen Kappa Score:', cohen_kappa)

# Classification Report
classification_report = metrics.classification_report(y_true, y_pred_classes)
print('\t\tClassification Report:\n', classification_report)


# In[21]:


from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches


# In[22]:


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])  # Assuming y_pred contains probabilities for class 1

# Calculate AUC score
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color='teal', lw=2, label='ROC curve (AUC = %0.5f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('VGG16 (ROC)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)

# Add the arrow to the plot
arrow = mpatches.FancyArrowPatch((0.5, 0.5), (0.8, 0.7), mutation_scale=100)
plt.annotate('', xy=(0.3, 0.7), xytext=(0.5, 0.5),
             arrowprops={'arrowstyle': '->', 'color': 'green'},
             fontsize=12, ha='left', va='bottom')

# Add text near the arrow
plt.text(0.35, 0.8, 'More accurate area', fontsize=9, color='black', ha='center', va='center')

plt.annotate('', xy=(0.7, 0.3), xytext=(0.5, 0.5),
             arrowprops={'arrowstyle': '->', 'color': 'red'},
             fontsize=12, ha='left', va='bottom')

# Add text near the arrow
plt.text(0.8, 0.2, 'Less accurate area', fontsize=9, color='black', ha='center', va='center')

plt.show()

# Print AUC score
print('ROC AUC Score is :', roc_auc)


# In[23]:


# Calculate Specificity (SP) for training set
specificity_train = confusion_mtx_train[0, 0] / (confusion_mtx_train[0, 0] + confusion_mtx_train[0, 1])

# Calculate Sensitivity (SE) for training set
sensitivity_train = confusion_mtx_train[1, 1] / (confusion_mtx_train[1, 0] + confusion_mtx_train[1, 1])

print('Specificity (True Negative Rate) for Training Set = %.4f' % specificity_train)
print('Sensitivity (True Positive Rate) for Training Set = %.4f' % sensitivity_train)


# In[24]:


# Calculate Specificity (SP) for testing set
specificity_test = confusion_mtx[0, 0] / (confusion_mtx[0, 0] + confusion_mtx[0, 1])

# Calculate Sensitivity (SE) for testing set
sensitivity_test = confusion_mtx[1, 1] / (confusion_mtx[1, 0] + confusion_mtx[1, 1])

print('Specificity (True Negative Rate) for Testing Set = %.4f' % specificity_test)
print('Sensitivity (True Positive Rate) for Testing Set = %.4f' % sensitivity_test)


# In[25]:


# Concatenate true labels and predicted labels for both training and testing sets
combined_true_labels = np.concatenate((y_train_true, y_true))
combined_pred_labels = np.concatenate((y_train_pred_classes, y_pred_classes))

# Calculate confusion matrix for the entire dataset
confusion_mtx_combined = confusion_matrix(combined_true_labels, combined_pred_labels)

# Calculate Specificity (SP) for the entire dataset
specificity_combined = confusion_mtx_combined[0, 0] / (confusion_mtx_combined[0, 0] + confusion_mtx_combined[0, 1])

# Calculate Sensitivity (SE) for the entire dataset
sensitivity_combined = confusion_mtx_combined[1, 1] / (confusion_mtx_combined[1, 0] + confusion_mtx_combined[1, 1])

print('Specificity (True Negative Rate) for Entire Dataset = %.4f' % specificity_combined)
print('Sensitivity (True Positive Rate) for Entire Dataset = %.4f' % sensitivity_combined)


# In[26]:


# Set a cool color palette
cool_palette = sns.color_palette("autumn", as_cmap=True)

# Plot the confusion matrix with adjusted font size and color palette
plt.figure(figsize=(5,4))
sns.set(font_scale=1.2)  # Adjust font scale
sns.heatmap(confusion_mtx_combined, annot=True, fmt='d', cmap=cool_palette,
            xticklabels=['OSCC', 'Normal'],
            yticklabels=['OSCC', 'Normal'])
plt.xlabel('Predicted', fontsize=14)  # Adjust label font size
plt.ylabel('True', fontsize=14)  # Adjust label font size
plt.title('Confusion Matrix (Entire Dataset)', fontsize=16)  # Adjust title font size
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

init_notebook_mode(connected=True)
RANDOM_SEED = 123
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# In[6]:


IMG_PATH = r"E:\CSE438\Project\100_processed\Processed-20231209T082531Z-001\Processed"
# split the data by train/val/test
for CLASS in os.listdir(IMG_PATH):
    if not CLASS.startswith('.'):
        IMG_NUM = len(os.listdir(IMG_PATH + CLASS))
        for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH + CLASS)):
            img = IMG_PATH + CLASS + '/' + FILE_NAME
            if n < 5:
                shutil.copy(img, 'TEST/' + CLASS.upper() + '/' + FILE_NAME)
            elif n < 0.8*IMG_NUM:
                shutil.copy(img, 'TRAIN/'+ CLASS.upper() + '/' + FILE_NAME)
            else:
                shutil.copy(img, 'VAL/'+ CLASS.upper() + '/' + FILE_NAME)


# In[7]:


image_dir = Path("E:\CSE438\Project\Processed\Combine")


# In[8]:


filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)


# In[9]:


[os.path.abspath(filepaths[0]) for filepaths[0] in filepaths ]


# In[10]:


images.Label.value_counts()


# In[11]:


train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)


# In[12]:


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    validation_split=0.2)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)


# In[19]:


TRAIN_DIR = r"E:\CSE438\Project\100_processed\splitted\Processed\train"
TEST_DIR = r"E:\CSE438\Project\100_processed\splitted\Processed\test"
VAL_DIR = r"E:\CSE438\Project\100_processed\splitted\Processed\val"
IMG_SIZE = (224,224)


# In[30]:


# use predefined function to load the image data into workspace
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)


# In[20]:


# TRAIN_DIR = 'TRAIN_CROP/'
# VAL_DIR = 'VAL_CROP/'

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)

validation_generator = train_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)


# In[32]:


vgg16_weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)


# In[21]:


pretrained_model = tf.keras.applications.vgg16.VGG16(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')

pretrained_model.trainable = False


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[23]:


NUM_CLASSES = 1

model = Sequential()
model.add(pretrained_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

model.summary()


# In[24]:


EPOCHS = 20
es = EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[es]
)


# In[25]:


# plot model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()


# In[26]:


results = model.evaluate(test_generator, verbose=0)
print(results)
print(f"Test Accuracy: {np.round(results[1] * 100,2)}%")


# In[51]:


results = model.evaluate(train_generator, verbose=0)
print(results)
print(f"Train Accuracy: {np.round(results[1] * 100,2)}%")


# In[52]:


train_loss, train_accuracy = model.evaluate(train_generator)
print(f"Training accuracy: {accuracy}")


# In[27]:


train_loss, train_accuracy = model.evaluate(train_generator)
print(f"Training accuracy: {train_accuracy}")


# In[28]:


predictions = np.argmax(model.predict(test_generator), axis=1)
matrix = confusion_matrix(test_generator.labels, predictions)
report= classification_report(test_generator.labels, predictions, target_names=test_generator.class_indices, zero_division=0)


# In[29]:


fig = plt.figure(figsize=(30, 30))
sns.heatmap(matrix, annot=True, cmap='viridis')
plt.xticks(ticks=np.arange(12) + 0, labels=test_generator.class_indices, rotation=90)
plt.yticks(ticks=np.arange(12) + 0.5, labels=test_generator.class_indices, rotation=0)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fig.savefig("Confusion Matrix",dpi=700)


# In[30]:


plt.figure(figsize=(38, 30))
ax = plt.subplot()
sns.set(font_scale=2.0)
sns.heatmap(matrix, annot=True, fmt='g', cmap="Blues", ax=ax); 

# labels, title and ticks
ax.set_xlabel('Predicted labels', fontsize=20);ax.set_ylabel('True labels', fontsize=10); 
ax.set_title('Confusion Matrix', fontsize=20); 
ax.xaxis.set_ticklabels(['Normal','OSCC'], fontsize=10);


# In[43]:


predictions = model.predict(X_val_prep)
predictions = [1 if x>0.5 else 0 for x in predictions]


accuracy = accuracy_score(y_val, predictions)
print('Val Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_val, predictions) 
cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)


# In[44]:


predictions = model.predict(X_test_prep)
predictions = [1 if x>0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_test, predictions)
print('Test Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_test, predictions) 
cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)


# In[38]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage import io 
import os

# Define your paths to the dataset
train_dir = r"E:\CSE438\Project\Processed\Splitter Cancer Non cancer\train"
validation_dir =  r"E:\CSE438\Project\Processed\Splitter Cancer Non cancer\val"

# Creating ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

dataset = []

import numpy as np 
from PIL import Image 

image_directiory = 
SIZE = 128 
dataset = []

my_image = os.listdir(image_directory)
for i, image_name in enumerate(my_image): 
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        
x = np.array(dataset)

i = 0 
for batch in datagen.flow(x, 
                         batch_size = 16,
                         save_to_dir = 'augmented',
                         save_prefix = 'aug',
                         save_format = 'png'):
    
    i+=1
    if i > 10:
        break

# Validation data should not be augmented
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow images in batches using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Displaying augmented images (optional)
# This part will show some augmented images
# Remove it if you don't want to display the images
augmented_images = train_generator[0][0]
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(augmented_images[i])
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




