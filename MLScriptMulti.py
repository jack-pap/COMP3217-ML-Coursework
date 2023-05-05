import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, precision_recall_curve

# Loads in training and testing data
trainData = pd.read_csv("TrainingDataMulti.csv")  
testData = pd.read_csv("TestingDataMulti.csv")

# Splits the training data into features and labels taking the last column as label for Y
X_training = trainData.iloc[:, :-1]
Y_training = trainData.iloc[:, -1]

print (X_training.shape)
print (Y_training.shape)

# Normalize the training data using a scaler
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)

# Splits data into training and validation 80% / 20% 
X_training, X_validation, Y_training, Y_validation = train_test_split(X_training, Y_training, test_size=0.15)

# Train a random forest classifier model on the training data
model = RandomForestClassifier()
model.fit(X_training, Y_training)

# Evaluate the model on the validation data set
Y_predictions = model.predict(X_validation)
Y_trainPredictions = model.predict(X_training)
accuracy = np.mean(Y_predictions == Y_validation)
f1 = f1_score(Y_validation, Y_predictions, average='macro')

print("Validation accuracy:", accuracy)
print ("F1 Score:",f1)

# Use the trained model to predict labels for the testing data
X_test = testData.values
X_test = scaler.transform(X_test)
y_pred_test = model.predict(X_test)

# Get confusion matrix for training data

cm = confusion_matrix(Y_training, Y_trainPredictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Reds)
plt.show()

# Get confusion matrix for validation data
cm = confusion_matrix(Y_validation, Y_predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.PuBu)
plt.show()

# Output test label predictions to csv file
testData['LabelPred'] = y_pred_test
testData.to_csv('TestingResultsBinary.csv', index=False)