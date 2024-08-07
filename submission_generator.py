import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

#load image helper function
def load_image_to_test(file_path):
    img = Image.open(file_path)
    img = img.resize((96, 96))
    img = np.array(img)/255.0
    return img

#load model and prepare files for evaluation
model = load_model('classifier_model.keras')

test_data_file_path = 'Data/test'

testing_files = [ i for i in os.listdir(test_data_file_path) if i.endswith('.tif')] 

#generate predictions on test dataset
predictions = {}

for file in testing_files:
    path = os.path.join(test_data_file_path, file)
    img = load_image_to_test(path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    predictions[file.split('.')[0]] = pred

#make data frame to convert to results csv file
df = pd.DataFrame.from_dict(predictions, orient='index', columns=['label'])
df.index.name = 'id'
df.to_csv('Submission.csv')

print('Complete!')


