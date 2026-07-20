import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("housing.csv")


print("Shape:")
print(df.shape)

print("\nInformation:")
print(df.info())

print("\nNull values:")
print(df.isnull().sum())

print("\nDescription:")
print(df.describe())


print("----------------------------------------------")


# Remove unnecessary text column
df = df.drop("Address", axis=1)


# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

print("DROPPPPPPPPPPPPPPPPPPPPPPPPPP")
print(X.columns)


print("Features:")
print(X)

print("\nTarget:")
print(y)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


print("\nPredicted prices:")
print(y_pred)

print("\nActual prices:")
print(y_test.values)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("----------------------------------------------")
print("Model Evaluation")
print("----------------------------------------------")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Model parameters
print("----------------------------------------------")
print("Coefficients:")
print(model.coef_)

print("Intercept:")
print(model.intercept_)

import pickle

pickle.dump(model, open("model.pkl", "wb"))