import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('output_traffic_data.csv')
data['current_date'] = pd.to_datetime(data['current_date'], format='%d-%m-%Y')
data['day_of_week'] = data['current_date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
data['current_time'] = pd.to_datetime(data['current_time'], format='%H:%M:%S')
data['hour_of_day'] = data['current_time'].dt.hour
data['minute_of_day'] = data['current_time'].dt.minute
data['duration_in_traffic_sec'] = data['duration_in_traffic'].apply(lambda x: int(x.split()[0]) * 60 if x != 'N/A' else 0)
data['distance_km'] = data['distance'].apply(lambda x: float(x.split()[0]))  # assuming distance is in km
data['origin_lat'] = data['origin_lat']
data['origin_lng'] = data['origin_lng']
data['destination_lat'] = data['destination_lat']
data['destination_lng'] = data['destination_lng']

features = ['day_of_week', 'hour_of_day', 'minute_of_day', 'duration_in_traffic_sec', 'distance_km', 
            'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng']
target_green = 'green_light'
target_red = 'red_light'

X = data[features]  
y_green = data[target_green] 
y_red = data[target_red]  
X_train, X_test, y_train_green, y_test_green = train_test_split(X, y_green, test_size=0.2, random_state=42)
X_train, X_test, y_train_red, y_test_red = train_test_split(X, y_red, test_size=0.2, random_state=42)
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, mse, r2


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")


# Linear Regression Model
linear_model_green = LinearRegression()
y_pred_green, mae_green, mse_green, r2_green = train_and_evaluate_model(linear_model_green, X_train, X_test, y_train_green, y_test_green)
save_model(linear_model_green, 'linear_model_green.pkl')

linear_model_red = LinearRegression()
y_pred_red, mae_red, mse_red, r2_red = train_and_evaluate_model(linear_model_red, X_train, X_test, y_train_red, y_test_red)
save_model(linear_model_red, 'linear_model_red.pkl')

# Random Forest Regressor Model
rf_model_green = RandomForestRegressor(n_estimators=100, random_state=42)
y_pred_rf_green, mae_rf_green, mse_rf_green, r2_rf_green = train_and_evaluate_model(rf_model_green, X_train, X_test, y_train_green, y_test_green)
save_model(rf_model_green,"rf_model_green.pkl")

rf_model_red = RandomForestRegressor(n_estimators=100, random_state=42)
y_pred_rf_red, mae_rf_red, mse_rf_red, r2_rf_red = train_and_evaluate_model(rf_model_red, X_train, X_test, y_train_red, y_test_red)
save_model(rf_model_red,"rf_model_red.pkl")


# XGBoost Model 
xgb_model_green = XGBRegressor(n_estimators=100, random_state=42)
y_pred_xgb_green, mae_xgb_green, mse_xgb_green, r2_xgb_green = train_and_evaluate_model(xgb_model_green, X_train, X_test, y_train_green, y_test_green)
save_model(xgb_model_green, 'xgb_model_green.pkl')

xgb_model_red = XGBRegressor(n_estimators=100, random_state=42)
y_pred_xgb_red, mae_xgb_red, mse_xgb_red, r2_xgb_red = train_and_evaluate_model(xgb_model_red, X_train, X_test, y_train_red, y_test_red)
save_model(xgb_model_red, 'xgb_model_red.pkl')

print("Green Signal Duration Model Evaluation:")
print(f"Linear Regression MAE: {mae_green:.4f}, MSE: {mse_green:.4f}, R2: {r2_green:.4f}")
print(f"Random Forest MAE: {mae_rf_green:.4f}, MSE: {mse_rf_green:.4f}, R2: {r2_rf_green:.4f}")
print(f"XGBoost MAE: {mae_xgb_green:.4f}, MSE: {mse_xgb_green:.4f}, R2: {r2_xgb_green:.4f}")

print("\nRed Signal Duration Model Evaluation:")
print(f"Linear Regression MAE: {mae_red:.4f}, MSE: {mse_red:.4f}, R2: {r2_red:.4f}")
print(f"Random Forest MAE: {mae_rf_red:.4f}, MSE: {mse_rf_red:.4f}, R2: {r2_rf_red:.4f}")
print(f"XGBoost MAE: {mae_xgb_red:.4f}, MSE: {mse_xgb_red:.4f}, R2: {r2_xgb_red:.4f}")
