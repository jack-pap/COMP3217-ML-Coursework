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

# Splits data into training and validation 70% / 30% 
X_training, X_validation, Y_training, Y_validation = train_test_split(X_training, Y_training, test_size=0.3)

# Train a random forest classifier model on the training data with a max tree depth for L1 regularization
model = RandomForestClassifier(max_depth=12) # 0.98 - 0.99 
#model = KNeighborsClassifier() # 0.90 - 0.94
#model = DecisionTreeClassifier() # 0.96- 0.98
model.fit(X_training, Y_training)

# Evaluate the model on the training data set
Y_trainPredictions = model.predict(X_training)
trainAccuracy = accuracy_score(Y_trainPredictions,Y_training)
trainF1 = f1_score(Y_trainPredictions, Y_training)
trainPrecision = precision_score(Y_trainPredictions, Y_training)
trainRecall = recall_score(Y_trainPredictions, Y_training)

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
validF1 = f1_score(Y_validation, Y_predictions)
validPrecision = precision_score(Y_validation, Y_predictions)
validRecall = recall_score(Y_validation, Y_predictions)

# Calculate the training error during the training process
valid_loss = mean_squared_error(Y_validation, Y_predictions)

print("\nValidation accuracy:", validAccuracy)
print ("Validation F1 Score:", validF1)
print("Validation Precision:", validPrecision)
print("Validation Recall:", validRecall)

print("Validation Error:", valid_loss)


# Splits the testing data into features and then normalizes it 
X_test = testData.values
print (X_test.shape)
X_test = scaler.transform(X_test)

# Use the trained model to predict labels for the testing data
y_pred_test = model.predict(X_test)

# Output precision recall curve for training data
precision, recall, thresholds = precision_recall_curve(Y_training, Y_trainPredictions)
plt.plot(recall, precision, color='r')
plt.xlabel('Training Recall')
plt.ylabel('Training Precision')
plt.title('Training Precision-Recall Curve')
plt.show()

# Output precision recall curve for validation data
precision, recall, thresholds = precision_recall_curve(Y_validation, Y_predictions)
plt.plot(recall, precision, color='b')
plt.xlabel('Validation Recall')
plt.ylabel('Validation Precision')
plt.title('Validation Precision-Recall Curve')
plt.show()

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
testData.to_csv('TestingResultsBinary.csv', index=False)