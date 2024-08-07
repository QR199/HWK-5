# HWK-5

This is my HWk-5 repository for the kaggle competition: "Histopathologic Cancer Detection"

File structure:

## HWK_5_model.py

This file contains the code I used to train my model, it also produced graphs for the final report.

## classifier_model.keras

This file stores the model trained by the code in HWK_5_model.py

## submission_generator

This file contains the code I used to make predictions on the entire test data set using the trained model. It also converted those predictions into a CSV file.

## Submission.csv

This file has the image id's along with their predicted probability of containing cancerous cells.

## Generated Images 

This is a folder that contains the images I created for the final report.

AUC.png: A graph of training and validation area under the curve throughout training.

Loss.png: A graph of the training and validation binary cross entropy loss throughout training.

Distribution_of_Labels.png: A histogram of the distribution of cancer positive or negative images.

Cancer_present.png: An example of an image that contains cancerous cells.

No_cancer_present.png: An example of an image that does not contain cancerous cells. 

Kaggle_score.png: A screenshot of my Kaggle submission score. 

## Libraries and Data:

This project used the libraries:

- Pandas
- Numpy
- matplotlib
- seaborn
- scikit-learn
- tensroflow
- keras
- os

The data for this project was manually downloaded from the Kaggle competition: Histopathologic Cancer Detection. 
It is not included in this repository because of its size, I also used a virtual environment native
to python (venv) to train the model.


