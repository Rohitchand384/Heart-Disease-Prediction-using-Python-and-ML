import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("heart.csv")

print(data)

print("*******************************************")

print(data.isnull().sum())

print("*******************************************")

print(data["RestingECG"].value_counts())
print(data["ChestPainType"].value_counts())
print(data["Sex"].value_counts())
print(data["ExerciseAngina"].value_counts())
print(data["ST_Slope"].value_counts())

print("*******************************************")

# RestingECG
resting_ecg_counts = [552, 188, 178]
resting_ecg_labels = ['Normal', 'LVH', 'ST']

plt.bar(resting_ecg_labels, resting_ecg_counts)
plt.xlabel('RestingECG')
plt.ylabel('Count')
plt.title('Resting ECG Distribution')
plt.show()

# ChestPainType
chest_pain_counts = [496, 203, 173, 46]
chest_pain_labels = ['ASY', 'NAP', 'ATA', 'TA']

plt.bar(chest_pain_labels, chest_pain_counts)
plt.xlabel('ChestPainType')
plt.ylabel('Count')
plt.title('Chest Pain Type Distribution')
plt.show()

# Sex
sex_counts = [725, 193]
sex_labels = ['M', 'F']

plt.pie(sex_counts, labels=sex_labels, autopct='%1.1f%%')
plt.title('Sex Distribution')
plt.show()

# ExerciseAngina
exercise_angina_counts = [547, 371]
exercise_angina_labels = ['N', 'Y']

plt.pie(exercise_angina_counts, labels=exercise_angina_labels, autopct='%1.1f%%')
plt.title('Exercise-Induced Angina Distribution')
plt.show()

# ST_Slope
st_slope_counts = [460, 395, 63]
st_slope_labels = ['Flat', 'Up', 'Down']

plt.bar(st_slope_labels, st_slope_counts)
plt.xlabel('ST_Slope')
plt.ylabel('Count')
plt.title('ST Slope Distribution')
plt.show()

print("*******************************************")

from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["ChestPainType"] = label_encoder.fit_transform(data["ChestPainType"])
data["RestingECG"] = label_encoder.fit_transform(data["RestingECG"])
data["ExerciseAngina"] = label_encoder.fit_transform(data["ExerciseAngina"])
data["ST_Slope"] = label_encoder.fit_transform(data["ST_Slope"])

# Print the encoded DataFrame
data.info()

print("*******************************************")

print (data)

print("*******************************************")

x=data.iloc[:,:11].values
y=data.iloc[:,11].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=1)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# Create an instance of each classifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)


classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}: Accuracy = {accuracy}')
    print("*******************************************")
    
print("\n*******************************************\n")

print("\nApplying Random Forest classifier\n")

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(x_test)


# Age, Sex,	ChestPainType,	RestingBP,	Cholesterol,	FastingBS,	RestingECG,	MaxHR,	ExerciseAngina, Oldpeak, ST_Slope

new_data = [[40, 1, 1, 140, 289, 0, 1, 172, 0, 0.0, 2]]
# Apply the same scaling factors as used for training data
new_data_scaled = sc.transform(new_data)

# Make predictions on the standardized input data
predictions = rf_classifier.predict(new_data_scaled)
if(predictions==0):
    print("No disease Predicted")
else:
    print("Disease Predicted")



    


