import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score

# -----------------------------
# Create Employee Dataset
# -----------------------------
data = {
    "Age":[25,30,45,35,29,41,38,27,50,32,28,40,36,31,26,42,37,33,39,29],
    "MonthlyIncome":[3000,4500,8000,6000,3500,9000,7000,3200,10000,5000,
                     3800,8500,6500,4800,3400,9200,7200,5100,7800,3600],
    "YearsAtCompany":[1,5,15,10,2,18,12,1,20,7,3,16,9,6,2,19,11,8,13,4],
    "JobSatisfaction":["Low","Medium","High","High","Low","High","Medium",
                       "Low","High","Medium","Low","High","Medium","Medium",
                       "Low","High","Medium","Medium","High","Low"],
    "OverTime":["Yes","No","No","Yes","Yes","No","No","Yes","No","Yes",
                "Yes","No","No","Yes","Yes","No","No","Yes","No","Yes"],
    "Attrition":["Yes","No","No","No","Yes","No","No","Yes","No","No",
                 "Yes","No","No","No","Yes","No","No","No","No","Yes"]
}

df = pd.DataFrame(data)

print("Employee Dataset")
print(df)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

for col in ["JobSatisfaction","OverTime","Attrition"]:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="AUC = %.2f" % auc)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()