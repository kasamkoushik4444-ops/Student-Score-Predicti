# Student-Score-Predicti
Student Exam Score Prediction is a simple machine learning project that predicts a student’s exam score using hours studied and attendance percentage. Linear Regression is used to train the model, evaluate performance, visualize trends, and make real-time predictions based on user input.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Attendance":   [60,65,70,75,80,85,90,95,96,98],
    "Score":        [35,40,45,50,55,65,70,80,85,92]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Attendance"]]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Evaluation Results")
print("------------------------")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

plt.scatter(df["Hours_Studied"], df["Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.show()

hours = float(input("Enter hours studied per day: "))
attendance = float(input("Enter attendance percentage: "))

prediction = model.predict([[hours, attendance]])
print("Predicted Exam Score:", round(prediction[0], 2))
