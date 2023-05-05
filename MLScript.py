import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics  import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

trainData = pd.read_csv("TrainingDataBinary.csv")  
testData = pd.read_csv("TestingDataBinary.csv")

# Splits the data into features and labels taking the last column as label for Y
X_training = trainData.iloc[:, :-1]
Y_training = trainData.iloc[:, -1]

X_testing = testData.iloc[:, :-1]
Y_testing = testData.iloc[:, -1]

print (X_training.shape)
print (Y_training.shape)

# Normalize the training data using a scaler
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)

# Split the data into training and validation sets 80/20
X_training, X_testing, Y_training, Y_testing = train_test_split(X_training, Y_training, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression(C= 20, max_iter=1000)
model.fit(X_training, Y_training)

# Evaluate the model on the testing data
accuracy = model.score(X_testing, Y_testing)
print("Validation accuracy:", accuracy)
predictions = model.predict(X_testing)
print ("F1 Score:",f1_score(Y_testing, predictions, average='macro'))

#get confusion matrix
cm = confusion_matrix(Y_testing, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
