# Step 1: Import libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# step 2: Load data
print("=="*30)
df=pd.read_csv("Rain_prediction.csv")
print("Original data :\n",df)
print("=="*30)

 #nSTEP 3: INITIAL VISUALIZATION (MISSING)
print("="*60)
print("INITIAL VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Temperature distribution
axes[0,0].hist(df["Temperature_C"].dropna(), bins=10, color='skyblue', edgecolor='black')
axes[0,0].set_title("Temperature Distribution")
axes[0,0].set_xlabel("Temperature (°C)")

# Humidity distribution
axes[0,1].hist(df["Humidity_Percent"].dropna(), bins=10, color='lightgreen', edgecolor='black')
axes[0,1].set_title("Humidity Distribution")
axes[0,1].set_xlabel("Humidity (%)")

# Wind Speed distribution
axes[0,2].hist(df["Wind_Speed_kmh"].dropna(), bins=10, color='salmon', edgecolor='black')
axes[0,2].set_title("Wind Speed Distribution")
axes[0,2].set_xlabel("Wind Speed (km/h)")

# Rain_Tomorrow class distribution
rain_counts = df["Rain_Tomorrow"].value_counts()
axes[1,0].bar(rain_counts.index, rain_counts.values, color=['lightblue', 'lightcoral'])
axes[1,0].set_title("Rain Tomorrow - Class Distribution")
axes[1,0].set_ylabel("Count")

# Correlation heatmap (numerical only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    im = axes[1,1].imshow(corr, cmap='coolwarm')
    axes[1,1].set_xticks(range(len(corr.columns)))
    axes[1,1].set_yticks(range(len(corr.columns)))
    axes[1,1].set_xticklabels(corr.columns, rotation=45)
    axes[1,1].set_yticklabels(corr.columns)
    axes[1,1].set_title("Feature Correlations")
    plt.colorbar(im, ax=axes[1,1])

# Box plot for outliers
df[numeric_cols].boxplot(ax=axes[1,2])
axes[1,2].set_title("Outlier Detection")
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()


# step 4: EDA
print("Basic statistic:\n",df.describe())
print("Missing values:\n",df.isnull().sum())
print("columns:",df.columns.tolist())
print("Basic infoL:",df.info())
print("=="*30)
# step 5: Filling missing values
df["Temperature_C"]=df["Temperature_C"].fillna(df["Temperature_C"].mean())
print("After filling missing values :\n",df)
print("=="*30)
# step 6: Conversion of high and low into numeric in humidity
df["Humidity_Percent"]=df["Humidity_Percent"].replace("high",85)
df["Humidity_Percent"]=df["Humidity_Percent"].replace("low",45)

df["Humidity_Percent"]=pd.to_numeric(df["Humidity_Percent"],errors="coerce")
# STEP 7: Binning /Buckting
# Temperature binning
bins_temp = [0, 15, 25, 35, 50]
labels_temp = ["Very Cold", "Cold", "Warm", "Hot"]
df["Temp_Bin"] = pd.cut(df["Temperature_C"], bins=bins_temp, labels=labels_temp)

# Humidity binning
bins= [0, 40, 60, 80, 100]
labels = ["Low", "Moderate", "High", "Very High"]
df["Humidity_Bin"] = pd.cut(df["Humidity_Percent"], bins=bins, labels=labels)

# Wind Speed binning
bins_wind = [0, 10, 20, 30, 100]
labels_wind = ["Calm", "Breeze", "Windy", "Stormy"]
df["Wind_Bin"] = pd.cut(df["Wind_Speed_kmh"], bins=bins_wind, labels=labels_wind)

print("Binning completed!")
print(df[["Temperature_C", "Temp_Bin", "Humidity_Percent", "Humidity_Bin", "Wind_Speed_kmh", "Wind_Bin"]].head())

# STEP 8: One-Hot-Encoder

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=["Temp_Bin", "Humidity_Bin", "Wind_Bin"], 
                            prefix=["Temp", "Humidity", "Wind"])

# Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded["Rain_Tomorrow_Encoded"] = le.fit_transform(df_encoded["Rain_Tomorrow"])

print("One-hot encoding completed!")
print(f"Original columns: {len(df.columns)}")
print(f"After encoding: {len(df_encoded.columns)}")

# step 9: Feature and target 
X = df_encoded[["Temperature_C","Humidity_Percent","Wind_Speed_kmh","Cloud_Cover","Pressure_hPa"]]

# Add all dummy columns from one-hot encoding
dummy_cols = [col for col in df_encoded.columns if col.startswith(("Temp_", "Humidity_", "Wind_"))]
X = pd.concat([X, df_encoded[dummy_cols]], axis=1)

# Use encoded target
Y = df_encoded["Rain_Tomorrow_Encoded"]

print("Features shape:", X.shape)
print("Target shape:", Y.shape)
# step 10: Train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

# Step 11: Feature scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_Scaled=scaler.transform(X_test)

# Step 12: Model selection
model=LogisticRegression()
model.fit(X_train_scaled,Y_train)

# step 13: Predictions and probability
y_pred=model.predict(X_test_Scaled)
y_prob=model.predict_proba(X_test_Scaled)
print("=="*45)
# step 14: Evaluation
print("Accuracy:",accuracy_score(Y_test,y_pred))
print("Classification report:",classification_report(Y_test,y_pred))
print("Confusion matrix:",confusion_matrix(Y_test,y_pred))
print("=="*45)

# STEP 15: FINAL VISUALIZATION (MISSING)
print("="*60)
print("FINAL VISUALIZATION - RESULTS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Confusion matrix visualisation
cm = confusion_matrix(Y_test, y_pred)
im = axes[0,0].imshow(cm, cmap='Blues')
axes[0,0].set_title("Confusion Matrix")
axes[0,0].set_xlabel("Predicted")
axes[0,0].set_ylabel("Actual")
axes[0,0].set_xticks([0,1])
axes[0,0].set_yticks([0,1])
axes[0,0].set_xticklabels(['No Rain', 'Rain'])
axes[0,0].set_yticklabels(['No Rain', 'Rain'])
for i in range(2):
    for j in range(2):
        axes[0,0].text(j, i, cm[i,j], ha="center", va="center")
plt.colorbar(im, ax=axes[0,0])

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=True)

axes[0,1].barh(feature_importance['Feature'], feature_importance['Coefficient'], color='steelblue')
axes[0,1].set_title("Feature Importance")
axes[0,1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Actual vs Predicted comparison
axes[1,0].scatter(range(len(Y_test)), Y_test, alpha=0.5, label='Actual', color='blue')
axes[1,0].scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predicted', color='red')
axes[1,0].set_title("Actual vs Predicted")
axes[1,0].set_xlabel("Sample")
axes[1,0].set_ylabel("Rain (0=No, 1=Yes)")
axes[1,0].legend()

# Prediction probabilities distribution
axes[1,1].hist(y_prob[:,1], bins=10, color='green', alpha=0.7)
axes[1,1].set_title("Prediction Probabilities Distribution")
axes[1,1].set_xlabel("Probability of Rain")
axes[1,1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Replace Step 16 with this:

# step 16: Predicting new values (FIXED)
# Create new data with realistic values
new_data = pd.DataFrame([[28, 75, 15, 4, 1012]], 
                        columns=["Temperature_C","Humidity_Percent","Wind_Speed_kmh","Cloud_Cover","Pressure_hPa"])

# Add dummy columns (set to 0 for new data)
for col in dummy_cols:
    new_data[col] = 0

# Ensure same column order as X
new_data = new_data[X.columns]

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print("prediction for tomorrow:", "Rain" if prediction[0]==1 else "No Rain")
print("=="*45)
print(f"Probability: {probability[0][1]*100:.2f}% chance of rain")
print("=="*45)


# step 17: Save model for Streamlit app
import joblib

# Save model and scaler
joblib.dump(model, "rain_model.pkl")
joblib.dump(scaler, "rain_scaler.pkl")

# Save feature columns (the ones you actually used in X)
joblib.dump(X.columns.tolist(), "rain_features.pkl")

# Save label encoder (to convert predictions back to text)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_encoded["Rain_Tomorrow"])  # Fit on original target
joblib.dump(le, "rain_label_encoder.pkl")

print("="*50)
print(" Model saved successfully!")
print("Files created:")
print("   - rain_model.pkl")
print("   - rain_scaler.pkl") 
print("   - rain_features.pkl")
print("   - rain_label_encoder.pkl")
print("="*50)
