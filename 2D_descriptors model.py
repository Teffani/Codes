import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors 


trainDataset = pd.read_excel(r"C:\Users\remin\Desktop\Dataset_1.xlsx")
testDataset = pd.read_excel(r"C:\Users\remin\Desktop\Dataset_2.xlsx")

#Generate canonical SMILES
def canonical_smiles(SMILES):
    mols = [Chem.MolFromSmiles(smi) for smi in SMILES] 
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return SMILES

# Canonical SMILES

Canon_SMILES_train = canonical_smiles(trainDataset.SMILES)
Canon_SMILES_test = canonical_smiles(testDataset.SMILES)


trainDataset['SMILES'] = Canon_SMILES_train
testDataset['SMILES'] = Canon_SMILES_test


Y = trainDataset['Tg (°C)'] 

def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 

# Function call
Mol_descriptorsTrain,desc_names = RDkit_descriptors(trainDataset['SMILES'])
Mol_descriptorsTest,desc_names = RDkit_descriptors(testDataset['SMILES'])

df_with_descriptors_cache_train = pd.DataFrame(Mol_descriptorsTrain,columns=desc_names)
df_with_descriptors = df_with_descriptors_cache_train.dropna(axis=1)
df_with_descriptors = df_with_descriptors.drop('Ipc', axis=1)
dataset = df_with_descriptors

df_with_descriptors_cache_test = pd.DataFrame(Mol_descriptorsTest,columns=desc_names)
df_with_descriptors_test = df_with_descriptors_cache_test.dropna(axis=1)
datasetTest = df_with_descriptors_test

X = dataset
X_test = datasetTest

# Instantiate the VarianceThreshold object with the specified threshold
selector = VarianceThreshold(threshold=0.09)

# Fit the selector to the dataset and transform it
dataset_cleaned = selector.fit_transform(dataset)

# Print the shape of the original and cleaned dataframes
print("Original dataframe shape:", dataset.shape)
print("Cleaned dataframe shape:", dataset_cleaned.shape)

scaler = StandardScaler()

X = dataset_cleaned
X_scaled = scaler.fit_transform(X)  
X = X_scaled

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_reg.fit(X_train, Y_train)
y_pred_rf = rf_reg.predict(X_test)

# XGBoost model evaluation
xgb_reg = XGBRegressor(n_estimators=1000, max_depth =3, learning_rate=0.05, random_state=42)
xgb_reg.fit(X_train, Y_train)
y_pred_xgb = xgb_reg.predict(X_test)


# Fit the models and get the R-squared values for the training and test data
train_r2_rf = rf_reg.score(X_train, Y_train)
train_r2_xgb = xgb_reg.score(X_train, Y_train)
test_r2_rf = r2_score(Y_test, y_pred_rf)
test_r2_xgb = r2_score(Y_test, y_pred_xgb)

# Predict using the trained models
y_pred_rf_test = rf_reg.predict(X_test)
y_pred_xgb_test = xgb_reg.predict(X_test)

# Calculate R-squared values for the test data
test_r2_rf_test = r2_score(Y_test, y_pred_rf_test)
test_r2_xgb_test = r2_score(Y_test, y_pred_xgb_test)


#Graph 1

# Plot the R-squared values for the test data
fig, ax = plt.subplots()
labels = ['Random Forest', 'XGBoost']
test_r2_values_test = [test_r2_rf_test, test_r2_xgb_test]
x = np.arange(len(labels))
width = 0.35
rects = ax.bar(x, test_r2_values_test, width, label='Test R2')

# Add text labels for the R-squared values
for rect, test_r2 in zip(rects, test_r2_values_test):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height, f"{test_r2:.3f}", ha='center', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0,1)
ax.set_title('Comparison of R-squared values for unseen test data')
plt.show()

#Graph 2

# Create a bar chart to compare the R-squared values
fig, ax = plt.subplots()
labels = ['Random Forest', 'XGBoost']
train_r2_values = [train_r2_rf, train_r2_xgb]
test_r2_values = [test_r2_rf, test_r2_xgb]
x = np.arange(len(labels))
width = 0.35
rects1 = ax.bar(x - width/2, train_r2_values, width, label='Training R2')
rects2 = ax.bar(x + width/2, test_r2_values, width, label='Test R2')

# Add text labels for the R-squared values
for rect1, rect2, train_r2, test_r2 in zip(rects1, rects2, train_r2_values, test_r2_values):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    ax.text(rect1.get_x() + rect1.get_width()/2., height1, f"{train_r2:.3f}", ha='center', va='bottom')
    ax.text(rect2.get_x() + rect2.get_width()/2., height2, f"{test_r2:.3f}", ha='center', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0,1)
ax.set_title('Comparison of R-squared values for training and test data 2D Descriptor')
plt.show()

mae_rf = mean_absolute_error(Y_test, y_pred_rf)
mse_rf = mean_squared_error(Y_test, y_pred_rf)
r2_rf = r2_score(Y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mape_rf = np.mean(np.abs((Y_test - y_pred_rf) / Y_test)) * 100
n = len(Y_test)
p = X_test.shape[1]
adjusted_r2_rf = 1 - (1 - r2_rf) * (n - 1) / (n - p - 1)

print("Random Forest Regression Metrics:")
print("Mean Absolute Error (MAE):", mae_rf)
print("Mean Squared Error (MSE):", mse_rf)
print("Root Mean Squared Error (RMSE):", rmse_rf)
print("R-squared (R2) Score:", r2_rf)
print("Mean Absolute Percentage Error (MAPE):", mape_rf)
print("Adjusted R-squared Score:", adjusted_r2_rf)

n = len(Y_test)
p = X_test.shape[1]
mae_xgb = mean_absolute_error(Y_test, y_pred_xgb)
mse_xgb = mean_squared_error(Y_test, y_pred_xgb)
r2_xgb = r2_score(Y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mape_xgb = np.mean(np.abs((Y_test - y_pred_xgb) / Y_test)) * 100
adjusted_r2_xgb = 1 - (1 - r2_xgb) * (n - 1) / (n - p - 1)

print("\nXGBoost Regression Metrics:")
print("Mean Absolute Error (MAE):", mae_xgb)
print("Mean Squared Error (MSE):", mse_xgb)
print("Root Mean Squared Error (RMSE):", rmse_xgb)
print("R-squared (R2) Score:", r2_xgb)
print("Mean Absolute Percentage Error (MAPE):", mape_xgb)
print("Adjusted R-squared Score:", adjusted_r2_xgb)
