import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns




#get our data in 'heart_data'
heart_data = pd.read_csv('heart.csv')

#print first lines
print(heart_data.head())

print(heart_data.describe())

#showing data (disease nbrs by gender)
sns.countplot(x="sex", hue="target", data=heart_data, palette="pastel")
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.legend(["No", "Yes"])
plt.show()

#splitting features 
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


X = X.drop_duplicates()
Y = Y[X.index]  # Align Y with the cleaned X



chosenModel=1



#X_train et X_test : Contiennent les caractéristiques pour l'entraînement et le test, respectivement.
#Y_train et Y_test : Contiennent les valeurs cibles associées pour l'entraînement et le test, respectivement.
#Stratification : On peut observer que les proportions de classes 0 et 1 sont respectées autant que possible dans les ensembles d'entraînement et de test.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression(max_iter=10000)

#training our data
model.fit(X_train, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("accuracy for logisticRegression is : ", test_data_accuracy)


if(chosenModel==2) :
    model = DecisionTreeClassifier(random_state=0, max_depth = 3)
    #training our data
    model.fit(X_train, Y_train)

    # accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print("accuracy for DecisionTreeClassifier is : ", test_data_accuracy)

elif(chosenModel==3) :
    model = RandomForestClassifier(criterion ='entropy', n_estimators= 300)
    #training our data
    model.fit(X_train, Y_train)

    # accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print("accuracy for RandomForestClassifier is : ", test_data_accuracy)  








############ try with an example


input_data = (34,0,1,118,210,0,1,192,0,0.7,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


# Use predict_proba to get the probabilities for each class
probabilities = model.predict_proba(input_data_reshaped)

# Print probabilities for each class
print("Probability of being healthy: {:.3f}".format(probabilities[0][0]))
print("Probability of having a heart disease: {:.3f}".format(probabilities[0][1]))

