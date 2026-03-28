print("===================== Customer churn predictions full pipeline ==========================")

# step 1: Import libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import joblib

print("=="*40)

# step 2: Load data with error handling
try:
    df = pd.read_csv("Customer_churn.csv")
    print("Data Load Successfully")
except FileNotFoundError:
    print("File not found, check file path!")
    exit()
except Exception as e:
    print("Error while loading data:", e)
    exit()

# step 3: Column validation
required_columns = [
    "tenure","monthly_charges","total_charges",
    "avg_monthly_gb_download","avg_calls_per_month",
    "customer_service_calls","contract_type",
    "paperless_billing","payment_method","churn"
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(" Missing columns:", missing_cols)
    exit()
else:
    print("All required columns present")

# step 4: Initial visualization
# churn distribution
df["churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# numerical feature distribution
df[["tenure","monthly_charges","total_charges"]].hist(figsize=(10,5))
plt.suptitle("Feature Distribution")
plt.show()

# step 5: EDA
print("Basic statistics:\n", df.describe())
print("=="*40)
print("Missing values:\n", df.isnull().sum())
print("=="*40)
print("Basic info:\n")
df.info()
print("=="*40)
print("Columns:\n", df.columns.tolist())
print("=="*40)

# step 6: One hot encoding
contract_encoded = pd.get_dummies(df["contract_type"], prefix="contract")
paperless_encoded = pd.get_dummies(df["paperless_billing"], prefix="billing")
payment_encoded = pd.get_dummies(df["payment_method"], prefix="payment")

df_encoded = df.drop(["customer_id","contract_type","paperless_billing","payment_method"], axis=1)
df_final = pd.concat([df_encoded, contract_encoded, paperless_encoded, payment_encoded], axis=1)

print("==="*40)
print("df final:\n", df_final.head())
print("==="*40)

# step 7: feature and target
X = df_final.drop("churn", axis=1)
Y = df["churn"]

# step 8: Convert target
Y = Y.map({'Yes': 1, 'No': 0})

# step 9: training columns
training_columns = X.columns.tolist()

# step 10: class balance check
print("Churn value distribution:\n", Y.value_counts(normalize=True))

# step 11: Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# step 12: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# step 13: Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, Y_train)

# step 14: Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# step 15: Shapes
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("=="*40)

# step 16: Evaluation
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", cm)

# step 17: Confusion Matrix Visual
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# step 18: ROC Curve
fpr, tpr, thresholds = roc_curve(Y_test, y_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC Score:", roc_auc)

# step 19: Model performance check
if roc_auc > 0.8:
    print("Strong model")
elif roc_auc > 0.7:
    print("Good model")
else:
    print("Needs improvement")

# step 20: Feature importance
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importance")
plt.show()

# step 21: Threshold tuning
threshold = 0.4
custom_pred = (y_prob[:,1] > threshold).astype(int)
print("Custom Threshold Accuracy:", accuracy_score(Y_test, custom_pred))

# step 22: New predictions
new_customers = pd.DataFrame([
    {
        'tenure': 8,'monthly_charges': 95.5,'total_charges': 764.0,
        'avg_monthly_gb_download': 32.5,'avg_calls_per_month': 58,
        'customer_service_calls': 3,'contract_type': 'Month-to-month',
        'paperless_billing': 'Yes','payment_method': 'Electronic check'
    }
])

def encode_new_customer(df):
    try:
        contract_encoded = pd.get_dummies(df["contract_type"], prefix="contract")
        paperless_encoded = pd.get_dummies(df["paperless_billing"], prefix="billing")
        payment_encoded = pd.get_dummies(df["payment_method"], prefix="payment")

        df_encoded = df.drop(["contract_type","paperless_billing","payment_method"], axis=1)
        df_final = pd.concat([df_encoded, contract_encoded, paperless_encoded, payment_encoded], axis=1)

        for col in training_columns:
            if col not in df_final.columns:
                df_final[col] = 0

        return df_final[training_columns]

    except Exception as e:
        print("Error:", e)
        return None

new_encoded = encode_new_customer(new_customers)

if new_encoded is not None:
    new_scaled = scaler.transform(new_encoded)
    pred = model.predict(new_scaled)
    prob = model.predict_proba(new_scaled)

    print("Prediction:", "Yes" if pred[0]==1 else "No")
    print("Churn Probability:", f"{prob[0][1]:.2%}")
else:
    print("Encoding failed")

# step 23: Save model
joblib.dump(model,"churn_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(training_columns,"training_columns.pkl")

print("Model saved successfully")
