import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import resample
import joblib

# Load data
data = pd.read_csv('../data/eng_sim_features.csv')

# Define features and targets
features = ['del_ToF', 'Kalman_ToF', 'del_RSSI', 'Kalman_RSSI']
targets = ['salinity (ppt)', 'total dissolved solids (g/L)']

X = data[features]
y = data[targets]

# Scaling and polynomial features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

mse_salt_percent = mse[0] / data['salinity (ppt)'].mean() * 100
mse_tds_percent = mse[1] / data['total dissolved solids (g/L)'].mean() * 100

mae_salt_percent = mae[0] / data['salinity (ppt)'].mean() * 100
mae_tds_percent = mae[1] / data['total dissolved solids (g/L)'].mean() * 100

print('salinity rmse: ', np.sqrt(mse[0]), ', ', mse_salt_percent, '%', ' of mean')
print('tds rmse: ', np.sqrt(mse[1]), ', ', mse_tds_percent, '%', ' of mean')
print()
print('salinity mae: ', mae[0], ', ', mae_salt_percent, '%', ' of mean')
print('tds mae: ', mae[1], ', ', mae_tds_percent, '%', ' of mean')
print()
print('salinity r2: ', r2[0])
print('tds r2: ', r2[1])

# Define custom scorer for RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Cross-validation scores for salinity (MAE and RMSE)
cv_mae_salt = cross_val_score(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
    X_poly, 
    y['salinity (ppt)'], 
    cv=5, 
    scoring='neg_mean_absolute_error'
)

cv_rmse_salt = cross_val_score(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
    X_poly, 
    y['salinity (ppt)'], 
    cv=5, 
    scoring=rmse_scorer
)

# Cross-validation scores for TDS (MAE and RMSE)
cv_mae_tds = cross_val_score(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
    X_poly, 
    y['total dissolved solids (g/L)'], 
    cv=5, 
    scoring='neg_mean_absolute_error'
)

cv_rmse_tds = cross_val_score(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
    X_poly, 
    y['total dissolved solids (g/L)'], 
    cv=5, 
    scoring=rmse_scorer
)

# Print cross-validation results
print('salinity CV MAE: ', -np.mean(cv_mae_salt))
print('tds CV MAE: ', -np.mean(cv_mae_tds))

print('salinity CV RMSE: ', -np.mean(cv_rmse_salt))
print('tds CV RMSE: ', -np.mean(cv_rmse_tds))

# Confidence intervals using bootstrapping
n_bootstraps = 20
salinity_preds = []
tds_preds = []

for i in range(n_bootstraps):
    X_resampled, y_resampled = resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred_bootstrap = model.predict(X_test)
    salinity_preds.append(y_pred_bootstrap[:, 0])
    tds_preds.append(y_pred_bootstrap[:, 1])

salinity_preds = np.array(salinity_preds)
tds_preds = np.array(tds_preds)

# Calculate 95% confidence intervals
salinity_lower_bound = np.percentile(salinity_preds, 2.5, axis=0)
salinity_upper_bound = np.percentile(salinity_preds, 97.5, axis=0)
tds_lower_bound = np.percentile(tds_preds, 2.5, axis=0)
tds_upper_bound = np.percentile(tds_preds, 97.5, axis=0)

print('Salinity 95% CI: ', salinity_lower_bound.mean(), '-', salinity_upper_bound.mean())
print('TDS 95% CI: ', tds_lower_bound.mean(), '-', tds_upper_bound.mean())

# Save the model and pre-processing objects
joblib.dump(model, 'saved_models/rfr_model.pxl')
joblib.dump(scaler, 'saved_models/scaler.pxl')
joblib.dump(poly, 'saved_models/poly_features.pxl')
