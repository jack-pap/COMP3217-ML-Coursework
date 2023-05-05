import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score

# Loads in training and testing data
trainData = pd.read_csv("TrainingDataBinary.csv")  
testData = pd.read_csv("TestingDataBinary.csv")

# Splits the training data into features and labels taking the last column as label for Y
X_training = trainData.iloc[:, :-1]
Y_training = trainData.iloc[:, -1]

print (X_training.shape)
print (Y_training.shape)

# Normalize the training data using a scaler
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)

X_training, X_testing, Y_training, Y_testing = train_test_split(X_training, Y_training, test_size=0.2)

# Train a random forest classifier model on the training data
model = RandomForestClassifier()
model.fit(X_training, Y_training)

# Evaluate the model on the testing data
Y_predictions = model.predict(X_testing)
accuracy = np.mean(Y_predictions == Y_testing)
f1 = f1_score(Y_testing, Y_predictions)
precision = precision_score(Y_testing, Y_predictions)
predictions = model.predict(X_testing)

print("Validation accuracy:", accuracy)
print ("F1 Score:",f1_score(Y_testing, predictions, average='macro'))
print("Precision:", precision)

# Use the trained model to predict labels for the testing data
X_test = testData.values
X_test = scaler.transform(X_test)
y_pred_test = model.predict(X_test)

#get confusion matrix
cm = confusion_matrix(Y_testing, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.PuBu)
plt.show()