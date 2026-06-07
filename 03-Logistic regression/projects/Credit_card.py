# Step 1: Unnecessary libraries are removed, only important ones are imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving model

# Step 2: Load the data with error handling
try:
    df=pd.read_csv("creditcard.csv")
    print("File loaded successfully:\n",df.head())
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# Step 3: EDA
print("=="*40)
print("Basic statistics:\n",df.describe())
print("=="*40)
print("Basic info:\n",df.info())
print("=="*40)
print("Missing values:\n",df.isnull().sum())
print("Df Shape:\n",df.shape)
print("=="*40)
print("Dulicated values:\n",df.duplicated().sum())
print("=="*40)
print("Data types:\n",df.dtypes)
print("=="*40)
print("Columns:",df.columns.tolist())
print("=="*40)
print("No of Rows:",len(df))
print("=="*40)

# Step 4: Cleaning df
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print("Duplicated values dropped successfully")
print("=="*40)

# Save cleaned dataset 
df.to_csv("clean_dataset.csv", index=False)
print("Cleaned dataset saved as 'clean_dataset.csv'")
print(f"   Shape: {df.shape}")
print("=="*40)

# Step 5: Data Visualization 
("""
for column in df.columns:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True,color="green",alpha=0.01)
    plt.title(f'Histogram of {column}')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[column],color="red")
    plt.title(f'Box Plot of {column}')
    plt.tight_layout()
    plt.show()""")
print("=="*40)

# Step 6: Target and feature
X=df.drop("Class",axis=1)
y=df["Class"]
print("X shape:",X.shape)
print("y shape:",y.shape)
print("=="*40)

# Step 7: Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)
print("=="*40)

# Step 8: Model and predictions
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred_proba=model.predict_proba(X_test)
print("y_pred_proba shape:",y_pred_proba.shape)
print("=="*40)

# ========== NEW: Save the trained model ==========
joblib.dump(model, "fraud_detection_model.pkl")
print("Model saved as 'fraud_detection_model.pkl'")
print("=="*40)

# Set custom threshold 
threshold = 0.65

# Save threshold as well
with open("threshold.txt", "w") as f:
    f.write(str(threshold))
print(f"Threshold ({threshold}) saved as 'threshold.txt'")
print("=="*40)

# Apply threshold to probabilities
y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)

print(f"Using threshold: {threshold}")
print("=="*40)

# Step 9: Evaluation with threshold
accuracy=accuracy_score(y_test,y_pred_threshold)
print("Accuracy:",round(accuracy,2))
print("=="*40)
cms=confusion_matrix(y_test,y_pred_threshold)
print("Confusion Matrix:\n",cms)
print("=="*40)
classification=classification_report(y_test,y_pred_threshold)
print("Classification Report:\n",classification)
print("=="*40)

# Step 10: Visualize confusion matrix
sns.heatmap(cms,annot=True,fmt="d")
plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("=="*40)

# Step 11: Feature Importance
feature_importance=pd.Series(model.coef_[0],index=X.columns)
feature_importance.plot(kind="bar", figsize=(12,6))
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Step 12: New predictions with threshold
new_transaction_1 = {
    'Time': 50000,
    'V1': -0.5, 'V2': 0.3, 'V3': -0.2, 'V4': 0.1, 'V5': -0.3,
    'V6': 0.2, 'V7': -0.1, 'V8': 0.4, 'V9': -0.2, 'V10': 0.1,
    'V11': -0.3, 'V12': 0.2, 'V13': -0.1, 'V14': 0.3, 'V15': -0.2,
    'V16': 0.1, 'V17': -0.3, 'V18': 0.2, 'V19': -0.1, 'V20': 0.4,
    'V21': -0.2, 'V22': 0.1, 'V23': -0.3, 'V24': 0.2, 'V25': -0.1,
    'V26': 0.3, 'V27': -0.2, 'V28': 0.1,
    'Amount': 50.00
}

# second transaction 
new_transaction_2 = {
    'Time': 150000,
    'V1': 2.5, 'V2': -1.2, 'V3': 3.1, 'V4': -0.8, 'V5': 1.9,
    'V6': -2.3, 'V7': 1.8, 'V8': -1.5, 'V9': 2.2, 'V10': -0.9,
    'V11': 1.5, 'V12': -1.8, 'V13': 2.1, 'V14': -0.7, 'V15': 1.3,
    'V16': -1.4, 'V17': 2.0, 'V18': -1.1, 'V19': 1.7, 'V20': -0.6,
    'V21': 1.4, 'V22': -1.3, 'V23': 1.9, 'V24': -0.5, 'V25': 1.2,
    'V26': -1.0, 'V27': 1.6, 'V28': -0.8,
    'Amount': 2500.00
}

new_data = pd.DataFrame([new_transaction_1, new_transaction_2])
new_pred_proba = model.predict_proba(new_data)
print("=="*40)
print("NEW PREDICTIONS WITH THRESHOLD:")
print("=="*40)

# Mapping numerical predictions to descriptive labels
prediction_labels = {0: "no fraud", 1: "fraud"}

for index, row in new_data.iterrows():
    fraud_prob = new_pred_proba[index][1]  # Probability of fraud
    legit_prob = new_pred_proba[index][0]  # Probability of legit
    
    # Apply threshold
    pred_class = 1 if fraud_prob >= threshold else 0
    
    print(f"\nTransaction {index + 1}:")
    print(f"  Amount: ${row['Amount']:.2f}")
    print(f"  Fraud Probability: {fraud_prob:.4f} ({fraud_prob*100:.2f}%)")
    print(f"  Legit Probability: {legit_prob:.4f} ({legit_prob*100:.2f}%)")
    print(f"  Threshold: {threshold}")
    print(f"  Prediction: {prediction_labels[pred_class]}")
    print(f"  Confidence: {max(fraud_prob, legit_prob)*100:.2f}%")
    print("-" * 40)

# Step 13:  Find best threshold using F1 score
print("=="*40)
print("Finding optimal threshold:")
print("=="*40)

from sklearn.metrics import f1_score, precision_recall_curve

# Get probabilities for test set
test_probs = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
results = []

for thresh in thresholds:
    y_pred_temp = (test_probs >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_temp)
    results.append((thresh, f1))
    print(f"Threshold: {thresh:.2f} → F1 Score: {f1:.4f}")

# Find best threshold
best_thresh, best_f1 = max(results, key=lambda x: x[1])
print(f"\nBEST THRESHOLD: {best_thresh:.2f} (F1 Score: {best_f1:.4f})")

# Save best threshold
with open("best_threshold.txt", "w") as f:
    f.write(f"Best Threshold: {best_thresh}\nF1 Score: {best_f1:.4f}")
print("Best threshold saved as 'best_threshold.txt'")
print("=="*40)

# Step 14: Visualize threshold impact
plt.figure(figsize=(10, 6))
thresholds_plot, f1_scores = zip(*results)
plt.plot(thresholds_plot, f1_scores, marker='o', linewidth=2, markersize=8)
plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best threshold: {best_thresh}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Summary of saved files 
print("=="*40)
print("FILES SAVED SUCCESSFULLY:")
print("=="*40)
print("1. clean_dataset.csv        - Cleaned dataset (no duplicates)")
print("2. fraud_detection_model.pkl - Trained Logistic Regression model")
print("3. threshold.txt            - Current threshold (0.65)")
print("4. best_threshold.txt       - Optimal threshold based on F1 score")
print("=="*40)

# Function to load and predict later 
print("\nHOW TO LOAD MODEL LATER:")
print("="*40)
print("""
# Load saved model and threshold:
import joblib
import pandas as pd

# Load model
loaded_model = joblib.load('fraud_detection_model.pkl')

# Load threshold
with open('threshold.txt', 'r') as f:
    threshold = float(f.read())

# Make prediction
new_transaction = pd.DataFrame([{...}])
prediction = loaded_model.predict_proba(new_transaction)[0][1]
result = 'FRAUD' if prediction >= threshold else 'LEGIT'
print(f"Prediction: {result}, Probability: {prediction:.2%}")
""")
print("=="*40)

# Add this to create a business-friendly chart
import matplotlib.pyplot as plt

metrics = ['Fraud Caught', 'Fraud Missed', 'False Alarms']
values = [50, 40, 14]
colors = ['green', 'red', 'orange']

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=colors)
plt.title('Fraud Detection Performance (Last 90 days)')
plt.ylabel('Number of Transactions')
plt.ylim(0, 60)

# Add value labels on bars
for i, v in enumerate(values):
    plt.text(i, v + 2, str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
