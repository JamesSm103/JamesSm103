import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # for regression
import joblib
# Load data
df = pd.read_csv("NL_Batting_2024.csv")
df['Tm'] = df['Tm'].astype('category')

# Define features and target
X = df.drop(columns=["OPS+", "SLG"])  # drop targets from features
y = df["OPS+"]  # or "SLG", or you can do multi-output separately

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define model
model = xgb.XGBRegressor(
    enable_categorical=True,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
)

# Train model
model.fit(X_train, y_train)
joblib.dump(model, "xgb_main_model.pkl")
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)