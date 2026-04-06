# Step 1:Import all important libraraies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# step 2: Load data wth error handling
print("=="*50)
try:
    df=pd.read_csv("toyota_stock_v2_features.csv")
    print("Data loaded successfully!")
    print("Original data :\n",df.head())
except FileNotFoundError:
    print("Error: File not found!")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()
print("=="*50)

# Step 3: EDA
print("Basic Statistic:\n",df.describe())
print("=="*50)
print("Missing values:\n",df.isnull().sum())
print("=="*50)
print("Basic Info :\n",df.info())
print("=="*50)
print("columns:",df.columns.tolist())
print("=="*50)
print("No of columns:\n",df.columns.value_counts())
print("=="*50)

# Step 4: Defining required columns
required_cols=['Volume','Return','MA_10', 'MA_50', 
               'Volatility_10', 'Volatility_30', 'RSI', 'MACD', 'MACD_Signal']
df=df[required_cols].dropna()

# step 5: Initail visualisation
import matplotlib.pyplot as plt

# List of features to plot
features_to_plot = ['Volume', 'RSI', 'MACD', 'MA_10', 'MA_50', 'Volatility_10', 'Volatility_30', 'MACD_Signal']

# Create a simple 2x4 grid
plt.figure(figsize=(20, 10))

for i, col in enumerate(features_to_plot):
    plt.subplot(2, 4, i + 1)
    # Using a sample to keep the plot responsive
    sample_df = df.sample(min(1000, len(df)))
    plt.scatter(sample_df[col], sample_df['Return'], alpha=0.5, color="#CC2EC4", s=10)
    plt.title(f'{col} vs Return', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('Return')
    plt.grid(True, alpha=0.2)

plt.suptitle('Simple Feature Scatter Analysis', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# step 6: Outlier detection(Using IQR method)
Q1=df["Return"].quantile(0.25)
Q3=df["Return"].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
print(f"Lower bound for outliers: {lower_bound}")
print(f"Upper bound for outliers: {upper_bound}")
outliers=df[(df["Return"]<lower_bound) | (df["Return"]>upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")


# step 7: Feature and target
X=df.drop("Return",axis=1)
y=df["Return"]

# step 8: Train test split (Using chronological split)

split_idx=int(len(df)*0.8)
X_train=X.iloc[:split_idx]
X_test=X.iloc[split_idx:]
y_train=y.iloc[:split_idx]
y_test=y.iloc[split_idx:]

print(f"Training set size:{len(X_train)}")
print(f"Test set size:{len(X_test)}")

# step 9: Standard scaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Step 10: Model selction and predictions
model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

# Step 11: Evalution

MSE=mean_squared_error(y_test,y_pred)
print("MSE:",MSE)
print("=="*50)
R2_score=r2_score(y_test,y_pred)
print("R2 score:",R2_score)
print("=="*50)

# step 12: Final visualisation
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred, alpha=0.5, color="#DB34CD", s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs Predicted Returns', fontsize=14)
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.grid(True, alpha=0.2)

# step 13: predicting new values
new_data=np.array([[1000000, 70, 1.5, 150, 145, 0.02, 0.03, 0.5]])
new_data_scaled=scaler.transform(new_data) 
predicted_return=model.predict(new_data_scaled)
print("Predicted Return for new data:", predicted_return[0])

# step 14: residual analysis
residuals=y_test-y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred,residuals, alpha=0.5, color="#12EE1DDA", s=10)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals vs Predicted Values', fontsize=14)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.2)


# step 15: Feature importance (Coefficients)
feature_importance=pd.Series(model.coef_,index=X.columns)
feature_importance=feature_importance.sort_values(ascending=False)  
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
plt.title('Feature Importance (Coefficients)', fontsize=14)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(True, alpha=0.2)
plt.show()

# step 16: Save the model  
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as 'linear_regression_model.pkl'")

