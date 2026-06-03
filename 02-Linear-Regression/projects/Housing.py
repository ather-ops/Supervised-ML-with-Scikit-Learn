print("=="*20,"House price pred project 1","=="*20)

# Step 1: Import all important Libearies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the data with error handling
try:
    df=pd.read_csv("Housing.csv")
    print("Data loaded successfully!:\n",df.head())
except FileNotFoundError:
    print("Error: The file 'Housing.csv' was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred while loading the data: {e}")

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
print("Columns:\n",df.columns.tolist())
print("=="*40)
print("No of Rows:",len(df))
print("=="*40)

# Step 4: Dropping unwanted columns
dropping_columns=["mainroad","guestroom","basement","hotwaterheating"]
df.drop(columns=dropping_columns, inplace=True)
print("Unwanted columns dropped successfully!")
print("=="*40)
print("Data after dropping columns:\n",df.head())
print("=="*40)

# Step 5: Manual encode the categorical columns
categorical_columns = ["airconditioning","prefarea","furnishingstatus"]
encoders = {}  # Dictionary to store all encoders

for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])
print("Categorical columns encoded successfully!")
print("=="*40)

# Check encodings (ye dekhna important hai)
print("Encoding mappings:")
for col in categorical_columns:
    mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
    print(f"  {col}: {mapping}")
print("=="*40)
# Step 6: Define X and y
X=df.drop(columns=["price"])
y=df["price"]
print("Features and target variable defined successfully!")
print("=="*40)
print("Features (X):\n",X.head())
print("=="*40)
print("Target variable (y):\n",y.head())
print("=="*40)

# Step 7: Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Train-test split completed successfully!")
print("=="*40)

# Step 8: Feature Scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print("Feature scaling completed successfully!")
print("=="*40)

# Step 9: Train the model
model=LinearRegression()
model.fit(X_train,y_train)
print("Model training completed successfully!")
print("=="*40)

# Step 10: Predictions
y_pred=model.predict(X_test)
print("Predictions made successfully!")
print("=="*40)
print("Predicted values:\n",y_pred[:5])
print("=="*40)

# Step 11: Evaluation
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Model evaluation completed successfully!")
print("=="*40)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print("=="*40)

# Step 12: Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test,y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y.min(),y.max()],[y.min(),y.max()],"r--")
plt.show()

# Step 13: Prediction for new data
print("Model features (in order):", X.columns.tolist())

# Step 13: Prediction for new data
new_house = {
    'area': 8000,
    'bedrooms': 3,
    'bathrooms': 2,
    'stories': 2,
    'airconditioning': 'yes',
    'parking': 2,
    'prefarea': 'yes',
    'furnishingstatus': 'furnished'
}

# Convert to array in correct order
new_data = np.array([[
    new_house['area'],
    new_house['bedrooms'],
    new_house['bathrooms'],
    new_house['stories'],
    encoders['airconditioning'].transform([new_house['airconditioning']])[0],
    new_house['parking'],
    encoders['prefarea'].transform([new_house['prefarea']])[0],
    encoders['furnishingstatus'].transform([new_house['furnishingstatus']])[0]
]])

new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)
print(f"Predicted price: ₹{new_prediction[0]:,.2f}")

# Step 14: Save the model
import joblib
joblib.dump(model,"house_price_model.pkl")
print("Model saved successfully!")
print("=="*40)


