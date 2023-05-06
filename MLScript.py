import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, precision_recall_curve, recall_score, accuracy_score

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

# Splits data into training and validation 80% / 20% 
X_training, X_validation, Y_training, Y_validation = train_test_split(X_training, Y_training, test_size=0.16)

# Train a random forest classifier model on the training data
model = RandomForestClassifier() # 0.98 - 0.99 
#model = KNeighborsClassifier() # 0.90 - 0.94
#model = DecisionTreeClassifier() # 0.96- 0.98
model.fit(X_training, Y_training)

# Evaluate the model on the training data set
Y_trainPredictions = model.predict(X_training)
accuracy = np.mean(Y_trainPredictions == Y_training)
f1 = f1_score(Y_trainPredictions, Y_training)
precision = precision_score(Y_trainPredictions, Y_training)
recall = recall_score(Y_trainPredictions, Y_training)

print("Training accuracy:", accuracy)
print ("Training F1 Score:", f1)
print("Training Precision:", precision)
print("Training Recall:", recall)

# Evaluate the model on the validation data set
Y_predictions = model.predict(X_validation)
accuracy = np.mean(Y_predictions == Y_validation)
f1 = f1_score(Y_validation, Y_predictions)
precision = precision_score(Y_validation, Y_predictions)
recall = recall_score(Y_validation, Y_predictions)

print("\nValidation accuracy:", accuracy)
print ("Validation F1 Score:", f1)
print("Validation Precision:", precision)
print("Validation Recall:", recall)

# Splits the testing data into features and then normalizes it 
X_test = testData.values
print (X_test.shape)
X_test = scaler.transform(X_test)

# Use the trained model to predict labels for the testing data
y_pred_test = model.predict(X_test)

# Output precision recall curve for training data
precision, recall, thresholds = precision_recall_curve(Y_training, Y_trainPredictions)
plt.plot(recall, precision, color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
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
testData['LabelPred'] = y_pred_test
testData.to_csv('TestingResultsBinary.csv', index=False)