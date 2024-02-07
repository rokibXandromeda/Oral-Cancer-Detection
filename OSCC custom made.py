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


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X[i], cmap="gray")
    #plt.imshow(X[i])
    plt.axis('off')
plt.show()


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print ("Shape of an image in X_train: ", X_train[0].shape)
print ("Shape of an image in X_test: ", X_test[0].shape)


# In[5]:


from sklearn import preprocessing  # Import the preprocessing module


# In[6]:


le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test)


# In[7]:


print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)


# In[8]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.layers import Activation, GlobalAveragePooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout

def improved_custom_CNN(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()

    # Convolutional Layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())  # Pooling layer
    model.add(BatchNormalization())  # Batch normalization

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())

#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(BatchNormalization())

    # Flatten for Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.5))  # Dropout for regularization

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Usage
model = improved_custom_CNN()


# In[9]:


print(model.summary())


# In[10]:


from tensorflow.keras.models import Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])





# In[11]:


history = model.fit(X_train,y_train,
                    epochs=10,
                    validation_data=(X_test,y_test),
                    verbose = 1,
                    initial_epoch=0)


# In[12]:


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


# In[13]:


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
confusion_mtx = confusion_matrix(y_true, y_pred_classes, labels=[0,1])

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


# In[14]:


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


# In[15]:


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


# In[16]:


from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches


# In[23]:


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
plt.title('Custome Made (ROC)', fontsize=14)
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


# In[18]:


# Calculate Specificity (SP) for training set
specificity_train = confusion_mtx_train[0, 0] / (confusion_mtx_train[0, 0] + confusion_mtx_train[0, 1])

# Calculate Sensitivity (SE) for training set
sensitivity_train = confusion_mtx_train[1, 1] / (confusion_mtx_train[1, 0] + confusion_mtx_train[1, 1])

print('Specificity (True Negative Rate) for Training Set = %.4f' % specificity_train)
print('Sensitivity (True Positive Rate) for Training Set = %.4f' % sensitivity_train)


# In[19]:


# Calculate Specificity (SP) for testing set
specificity_test = confusion_mtx[0, 0] / (confusion_mtx[0, 0] + confusion_mtx[0, 1])

# Calculate Sensitivity (SE) for testing set
sensitivity_test = confusion_mtx[1, 1] / (confusion_mtx[1, 0] + confusion_mtx[1, 1])

print('Specificity (True Negative Rate) for Testing Set = %.4f' % specificity_test)
print('Sensitivity (True Positive Rate) for Testing Set = %.4f' % sensitivity_test)


# In[20]:


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


# In[21]:


# Set a cool color palette
cool_palette = sns.color_palette("autumn", as_cmap=True)

# Plot the confusion matrix with adjusted font size and color palette
plt.figure(figsize=(5,4))
sns.set(font_scale=1.2)  # Adjust font scale
sns.heatmap(confusion_mtx_combined, annot=True, fmt='d', cmap=cool_palette,
            xticklabels=['HC', 'SCZ'],
            yticklabels=['HC', 'SCZ'])
plt.xlabel('Predicted', fontsize=14)  # Adjust label font size
plt.ylabel('True', fontsize=14)  # Adjust label font size
plt.title('Confusion Matrix (Entire Dataset)', fontsize=16)  # Adjust title font size
plt.show()


# In[ ]:





# In[ ]:




