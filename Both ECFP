import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer 
from sklearn.feature_selection import VarianceThreshold

from xgboost import XGBRegressor
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors 
from rdkit.Chem import Descriptors3D
from tqdm import tqdm

#ECFP_1024

df = pd.read_excel(r"C:\Users\remin\Desktop\Dataset_1.xlsx")

# generating ECFPs (also known as morgan fps) with 1,024 bit length
ECFP4_1024 = []
for i in range (0, len(df)):
    mol = Chem.MolFromSmiles(df.iloc[i,0]) 
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useFeatures=True)
    fp_list = np.unique(fp, return_inverse=True)[1].tolist()
    ECFP4_1024.append(fp_list)

ECFP4_1024 = pd.DataFrame(data= ECFP4_1024)
ECFP4_1024 = pd.concat([ECFP4_1024, df.iloc[:, -1]], axis = 1)

# generating ECFPs (also known as morgan fps) with 2,024 bit length
ECFP4_2024 = []
for i in range (0, len(df)):
    mol = Chem.MolFromSmiles(df.iloc[i,0]) 
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2024,useFeatures=True)
    fp_list = np.unique(fp, return_inverse=True)[1].tolist()
    ECFP4_2024.append(fp_list)

ECFP4_2024 = pd.DataFrame(data= ECFP4_2024)
ECFP4_2024 = pd.concat([ECFP4_2024, df.iloc[:, -1]], axis = 1)

dataset = ECFP4_1024

#preprocessing  
Y = dataset['Tg (°C)']
dataset = dataset.drop(['Tg (°C)'], axis=1)

scaler = StandardScaler()

# Scale the dataset using StandardScaler
X = dataset
X_scaled = scaler.fit_transform(X)  
X = X_scaled


print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for Random Forest Regressor
param_grid_rf = {
    'n_estimators': [100, 250, 500, 750, 1000],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize the Random Forest Regressor model
rf_reg = RandomForestRegressor(random_state=42)

# Perform grid search using 5-fold cross validation with progress indicator
with tqdm(total=100, desc="Grid Search RF", position=0) as pbar:
    grid_search_rf = GridSearchCV(estimator=rf_reg, param_grid=param_grid_rf, cv=5,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search_rf.fit(X_train, Y_train)
    pbar.update(20)

# Fit the grid search model to the training data
grid_search_rf.fit(X_train, Y_train)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:", grid_search_rf.best_params_)

# Initialize the Random Forest Regressor model with the best parameters
rf_reg_best = RandomForestRegressor(**grid_search_rf.best_params_, random_state=42)

# Fit the model to the training data
rf_reg_best.fit(X_train, Y_train)

# Generate predictions on the testing data
y_pred_rf_best = rf_reg_best.predict(X_test)

# Calculate the evaluation metrics
n = len(Y_test)
p = X_test.shape[1]
mae_rf_best = mean_absolute_error(Y_test, y_pred_rf_best)
mse_rf_best = mean_squared_error(Y_test, y_pred_rf_best)
r2_rf_best = r2_score(Y_test, y_pred_rf_best)
rmse_rf_best = np.sqrt(mse_rf_best)
mape_rf_best = np.mean(np.abs((Y_test - y_pred_rf_best) / Y_test)) * 100
adjusted_r2_rf_best = 1 - (1 - r2_rf_best) * (n - 1) / (n - p - 1)

print("Mean Absolute Error: ", mae_rf_best)
print("Mean Squared Error: ", mse_rf_best)
print("Root Mean Squared Error: ", rmse_rf_best)
print("Mean Absolute Percentage Error: ", mape_rf_best)
print("R-squared Score: ", r2_rf_best)
print("Adjusted R-squared Score: ", adjusted_r2_rf_best)

#GRID SEARCH XGB

param_grid = {
    'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': [3, 4, 5, 8],
    'n_estimators': [100, 250, 500, 750, 1000, 2500]
}

# Initialize the XGBRegressor model
xgb_reg = XGBRegressor(random_state=42)

# Perform grid search using 5-fold cross validation
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5,
                           scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit the grid search model to the training data
grid_search.fit(X_train, Y_train)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:", grid_search.best_params_)

# Initialize the XGBRegressor model with the best parameters
xgb_reg_best = XGBRegressor(**grid_search.best_params_, random_state=42)

# Fit the model to the training data
xgb_reg_best.fit(X_train, Y_train)

# Generate predictions on the testing data
y_pred_xgb_best = xgb_reg_best.predict(X_test)

# Calculate the evaluation metrics
n = len(Y_test)
p = X_test.shape[1]
mae_xgb_best = mean_absolute_error(Y_test, y_pred_xgb_best)
mse_xgb_best = mean_squared_error(Y_test, y_pred_xgb_best)
r2_xgb_best = r2_score(Y_test, y_pred_xgb_best)
rmse_xgb_best = np.sqrt(mse_xgb_best)
mape_xgb_best = np.mean(np.abs((Y_test - y_pred_xgb_best) / Y_test)) * 100
adjusted_r2_xgb_best = 1 - (1 - r2_xgb_best) * (n - 1) / (n - p - 1)

print("Mean Absolute Error: ", mae_xgb_best)
print("Mean Squared Error: ", mse_xgb_best)
print("Root Mean Squared Error: ", rmse_xgb_best)
print("Mean Absolute Percentage Error: ", mape_xgb_best)
print("R-squared Score: ", r2_xgb_best)
print("Adjusted R-squared Score: ", adjusted_r2_xgb_best)
