import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

train_data_labels = pd.read_csv('Data/train_labels.csv')

#EDA label distributions
sns.countplot(x='label', data=train_data_labels)
plt.title('Distribution of Labels')
plt.show()

#helper function to show images of cells
def display_image(id_image, label, folder = 'Data/train'):
    plt.figure(figsize=(10, 10))
    for i, id in enumerate(id_image):
        img_path = os.path.join(folder, f"{id}.tif")
        img = Image.open(img_path)
        plt.subplot(5, 5,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f'Label: {label}', fontsize=10)
    plt.show()

#label 0 first image
display_image(train_data_labels[train_data_labels['label']==0]['id'].values[:1], 0)

#label 1 first image
display_image(train_data_labels[train_data_labels['label']==1]['id'].values[:1], 1)

#helper function to load and normalize images
img_height, img_width = 96, 96

def load_image(id_list, folder='Data/train'):
    images = []
    for id in id_list:
        img_path = os.path.join(folder, f"{id}.tif")
        img = Image.open(img_path)
        img = img.resize((img_height, img_width))
        img = np.array(img)/255
        images.append(img)
    return np.array(images) 

#fraction of data  used to train and test model with train_test_split
test_fraction = 0.2
train_ids, val_ids, train_labels, val_labels = train_test_split(train_data_labels['id'], train_data_labels['label'], test_size=0.2, random_state=42)

sample_training_ids = np.random.choice(train_ids, size=int(len(train_ids) * test_fraction), replace=False)
sample_validation_ids = np.random.choice(val_ids, size=int(len(val_ids) * test_fraction), replace=False)

training_sample = load_image(sample_training_ids)
validation_sample = load_image(sample_validation_ids)

training_labels = train_data_labels.set_index('id').loc[sample_training_ids].values.flatten()
validation_labels = train_data_labels.set_index('id').loc[sample_validation_ids].values.flatten()

#Convolutional neural net classifier
model = Sequential([

    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#train and save model
learning_rate = Adam(learning_rate=0.001)

model.compile(optimizer=learning_rate, loss='binary_crossentropy', metrics=['auc'])

history = model.fit(training_sample, training_labels, epochs=5, batch_size=16, validation_data=(validation_sample, validation_labels))

model.save('classifier_model.keras')

#show plot of binary cross entropy loss in training and validation data 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#show plot of AUC in training and validation 
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()