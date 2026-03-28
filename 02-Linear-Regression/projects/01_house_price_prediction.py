import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("housing_sample.csv.txt")
print("="*50)
print("original data:\n", df.head(10))

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.scatter(df["Area"], df["Price"], color="red")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("AREA VS PRICE(Before Training)")

plt.subplot(1,3,2)
plt.scatter(df["Bedrooms"], df["Price"], color="yellow")
plt.xlabel("No of bedrooms")
plt.ylabel("Price")
plt.title("BEDROOMS VS PRICE(Before Training)")

plt.subplot(1,3,3)
plt.scatter(df["Age"], df["Price"], color="green")
plt.xlabel("Age")
plt.ylabel("Price")
plt.title("AGE VS PRICE (Before Training)")

plt.tight_layout()
plt.show()

print("Basic statistic:\n", df.describe())
print("missing values:\n", df.isnull().sum())
print("="*50)

Q1=df["Price"].quantile(0.25)
Q2=df["Price"].quantile(0.75)
IQR=Q2-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q2+1.5*IQR
print("lower_bound:",lower_bound)
print("upper_bound:",upper_bound)
print("="*50)

bins=[0,1000,3000,5000]
labels=["small_house", "medium_house", "large_house"]
df["Area_bins"] = pd.cut(df["Area"], bins=bins, labels=labels)
print(df[["Area", "Area_bins"]])
print("="*50)

df["mean_price_by_bin"]=df.groupby("Area_bins")["Price"].transform("mean")
print(df[["Area_bins","Price","mean_price_by_bin"]])
print("="*50)

bins=[0,3,6,10]
labels=["New_house","Moderate","old_house"]
df["Age_bins"]=pd.cut(df["Age"],bins=bins,labels=labels)
print(df[["Area_bins","Age","Age_bins"]])
print("="*50)

one_hot_area_bins=pd.get_dummies(df["Area_bins"])
one_hot_age_bins=pd.get_dummies(df["Age_bins"])
df=pd.concat([df,one_hot_area_bins,one_hot_age_bins],axis=1)
print("\nDataFrame after all one-hot encoding:")
print("="*150)
print(df.head(10))

X=df[["Area","Bedrooms","Age"]]
Y=df["Price"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled,Y_train)

y_pred=model.predict(X_test_scaled) 
print("="*150)
print("X_train shape:",X_train.shape)
print("X test shape:",X_test.shape)
print("Y train shape:",Y_train.shape)
print("Y test shape:",Y_test.shape)
print("y pred shape:",y_pred.shape)
MSE=mean_squared_error(Y_test,y_pred)
MAE=mean_absolute_error(Y_test,y_pred)
R2_SCORE=r2_score(Y_test,y_pred)
print(f"MSE:{MSE:.2f}, MAE:{MAE:.2f}, R2 SCORE:{R2_SCORE:.2f}")
print("="*50)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(X_test["Area"],Y_test,label="Actual line",color="red")
plt.plot(X_test["Area"],y_pred,label="Predicted line",color="green")
plt.title("AREA VS PRICE(After Training)")
plt.xlabel("AREA")
plt.ylabel("PRICE")
plt.grid(True)
plt.legend()
plt.subplot(1,3,2)
plt.scatter(X_test["Bedrooms"],Y_test,color="red",label="Actual line")
plt.plot(X_test["Bedrooms"],y_pred,label="Predicted line",color="green")
plt.xlabel("NO OF BEDROOMS")
plt.ylabel("PRICE")
plt.title("BEDROOMS VS PRICE (After Training)")
plt.grid(True)
plt.subplot(1,3,3)
plt.scatter(X_test["Age"],Y_test,color="red",label="Actual line")
plt.plot(X_test["Age"],y_pred,label="Predicted line",color="green")
plt.xlabel("AGE")
plt.ylabel("PRICE")
plt.title("AGE VS PRICE (After Training)")
plt.grid(True)
plt.tight_layout()
print("="*100)

new_house=np.array([[5600,6,2]])
new_house_scaled=scaler.transform(new_house)
predicted_price=model.predict(new_house_scaled)
print(f"predicted price of new house  is :{predicted_price[0]:.2f}")
print("="*100)

plt.figure()
plt.plot(Y_test,y_pred)
plt.xlabel("Actual price")
plt.ylabel("Predicted price ")
plt.title("ACTUAL VS PREDICTED PRICE")
plt.grid(True)
residuals=Y_test-y_pred
plt.figure(figsize=(6,5))
plt.hist(residuals,bins=10,color="yellow")
plt.xlabel("Residuals error")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.grid(True)
plt.show()
