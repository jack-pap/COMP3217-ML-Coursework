import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, precision_recall_curve, recall_score, accuracy_score, mean_squared_error

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
X_training, X_validation, Y_training, Y_validation = train_test_split(X_training, Y_training, test_size=0.2)

# Train a random forest classifier model on the training data with a max tree depth for L1 regularization
model = RandomForestClassifier(max_depth=12) # 0.95 - 0.96
#model = KNeighborsClassifier() # 0.85 - 0.89
#model = DecisionTreeClassifier() # 0.89- 0.92
model.fit(X_training, Y_training)

# Evaluate the model on the training data set
Y_trainPredictions = model.predict(X_training)
trainAccuracy = accuracy_score(Y_trainPredictions,Y_training)
trainF1 = f1_score(Y_trainPredictions, Y_training, average='micro')
trainPrecision = precision_score(Y_trainPredictions, Y_training, average='micro')
trainRecall = recall_score(Y_trainPredictions, Y_training, average='micro')

# Calculate the training error during the training process
train_loss = mean_squared_error(Y_training, Y_trainPredictions)

print("Training accuracy:", trainAccuracy)
print ("Training F1 Score:", trainF1)
print("Training Precision:", trainPrecision)
print("Training Recall:", trainRecall)

print("Train Error:", train_loss)

# Evaluate the model on the validation data set
Y_predictions = model.predict(X_validation)
validAccuracy = np.mean(Y_predictions == Y_validation)
validF1 = f1_score(Y_validation, Y_predictions, average='micro')
validPrecision = precision_score(Y_validation, Y_predictions, average='micro')
validRecall = recall_score(Y_validation, Y_predictions, average='micro')

# Calculate the training error during the training process
valid_loss = mean_squared_error(Y_validation, Y_predictions)

print("\nValidation accuracy:", validAccuracy)
print ("Validation F1 Score:", validF1)
print("Validation Precision:", validPrecision)
print("Validation Recall:", validRecall)

print("Validation Error:", valid_loss)

# Normalize testing data using a scaler
X_test = testData.values
print (X_test.shape)
X_test = scaler.transform(X_test)

# Use the trained model to predict labels for the testing data
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
testData['LabelPrediction'] = y_pred_test
testData.to_csv('TestingResultsMulti.csv', index=False)