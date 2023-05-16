import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestRegressor 
 
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


# read the Excel file
input_file = r"C:\Users\remin\Desktop\Dataset_1.xlsx"
inputData = pd.read_excel(input_file, dtype=str)
cleaned = inputData.replace(regex=r'\*', value ="")

# save the modified data to a new Excel file
output_file = r"C:\Users\remin\Desktop\Dataset_1_cleaned.csv"
cleaned.to_csv(output_file, index=False)

# read the new Excel file and display the first five rows of the data
dataset = pd.read_csv(output_file)

failed_smiles = ["CCCCOC(C1(C)C(=O)OC(=O)C1()C)C", "c1ccc(nc1)Sc1ccc(cn1)n1c(=O)c2c(c1=O)cc1c(c2)c(=O)n(c1=O)","CCCCCCCCCn1c2ccc(cc2c2c1cccc2)C=C(C(=O)OCCCCCCOC(=O)C(=Cc1ccc2c(c1)c1ccccc1n2)C#N)C#N","CCCCCCCCCCCCn1c()ccc1c1ccc(s1)c1ccc(s1)c1ccc(s1)","CCCOC(=O)c1ccc(cc1)OP1(=NP(=NP(=N1)()Oc1ccc(cc1)C(=O)OCCC)(Oc1ccc(cc1)C(=O)OCCC)Oc1ccc(cc1)C(=O)OCCC)Oc1ccc(cc1)C(C(F)(F)F)(C(F)(F)F)c1ccc(cc1)O","Oc1ccc(cc1)C(=O)Nc1ccc(cc1)Cc1ccc(cc1)NC(=O)c1ccc(cc1)Oc1nc()nc(n1)Sc1ccccc1"," O[Si](O[Si](O[Si](O[Si](O[Si](O[Si](Oc1c()c2ccccc2c2c1cccc2)(C)C)(C)C)(C)C)(C)C)(C)C)(C)C","CC/C=C/", "CC/C=C\\", "CCC/C=C/", "CC/C=C(/C(C)(C)C)\\", "COC(=O)C(/C=C(\C)/)(C)C", "CC1()CCCCC1","c1ccc(cc1)Oc1ccc(cc1)Oc1ccc(c2c1cccc2)c1ccc(c2c1cccc2)Oc1ccc(cc1)Oc1ccc(cc1)C1()OC(=O)c2c1cccc2",
        'O[Zn]OC(=O)c1ccccc1C(=O)OCCCCCOC(=O)Nc1cc(ccc1C)NC(=O)OCCCCCOC(=O)c1ccccc1C(=O)',
       'CCCCCOC(=O)c1ccccc1C(=O)O[Pb]OC(=O)c1ccccc1C(=O)OCCCCCOC(=O)NCCCCCCNC(=O)O',
       'O[Ca]OC(=O)c1ccccc1C(=O)OCCCCCOC(=O)NCCCCCCNC(=O)OCCCCCOC(=O)c1ccccc1C(=O)',
       'CCCCCCCCc1cc(sc1c1ccc(s1)c1ccc(s1)c1sc(cc1CCCCCCCC)c1ccc(s1))c1ccc(s1)[Si]([Si]([Si]([Si]([Si]([Si]([Si]([Si](C)(C))(C)C)(C)C)(C)C)(C)C)(C)C)(C)C)(C)C',
       'c1ccc(s1)c1ccc(s1)c1ccc(s1)[Si](C)(C)',
       'CCCC[Si]([Si](CCCC)(CCCC))(c1ccc(s1))CCCC',
       'O[Si](O[Si](O[Si](c1ccc(cc1)Oc1ccc(cc1)[Si](C)(C))(C)C)(C)C)(C)C',
       'O[Si](COC(=O)CCCCCCCCC(=O)Oc1ccc(cc1)S(=O)(=O)c1ccc(cc1)Oc1ccc(cc1)C=C1CCCC(=Cc2ccc(cc2)Oc2ccc(cc2)S(=O)(=O)c2ccc(cc2)OC(=O)CCCCCCCCC(=O)OC[Si](C)(C))C1=O)(C)C',
       'C=C([As](c1ccccc1))c1ccccc1' ]
dataset = dataset[~dataset['SMILES'].isin(failed_smiles)]

pd.options.display.max_rows = None

# Define a function to generate 3D descriptors

def generate_3d_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Calculate 3D descriptors
    descriptors = {}
    descriptors['Asphericity'] = Descriptors3D.Asphericity(mol)
    descriptors['Eccentricity'] = Descriptors3D.Eccentricity(mol)
    descriptors['InertialShapeFactor'] = Descriptors3D.InertialShapeFactor(mol)
    descriptors['NPR1'] = Descriptors3D.NPR1(mol)
    descriptors['NPR2'] = Descriptors3D.NPR2(mol)
    descriptors['PMI1'] = Descriptors3D.PMI1(mol)
    descriptors['PMI2'] = Descriptors3D.PMI2(mol)
    descriptors['PMI3'] = Descriptors3D.PMI3(mol)
    descriptors['RadiusOfGyration'] = Descriptors3D.RadiusOfGyration(mol)
    descriptors['SpherocityIndex'] = Descriptors3D.SpherocityIndex(mol)

    return descriptors

dataset['3D_descriptors'] = dataset['SMILES'].apply(generate_3d_descriptors)
dataset = pd.concat([dataset.drop(['3D_descriptors'], axis=1), dataset['3D_descriptors'].apply(pd.Series)], axis=1)


#preprocessing  
Y = dataset['Tg (°C)']
dataset = dataset.drop(["SMILES", 'Tg (°C)'], axis=1)

# Instantiate the VarianceThreshold object with the specified threshold
selector = VarianceThreshold(threshold=0.2)

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

#GRAPH 1

# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(xgb_reg_best, X_train, Y_train, cv=5,
                                                        scoring='neg_mean_absolute_error',
                                                        n_jobs=-1, shuffle=True,
                                                        random_state=42)
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
plt.legend(loc="best")
plt.title("XGBoost learning curve")
plt.xlabel("Training examples")
plt.ylabel("Mean Absolute Error")
plt.show()

#GRAPH 2

# Get the mean test scores and standard deviations for each iteration
mean_test_scores = np.array(grid_search.cv_results_['mean_test_score'])
std_test_scores = np.array(grid_search.cv_results_['std_test_score'])

# Define a colormap with a sufficient number of colors
colors = plt.cm.get_cmap('tab20', len(mean_test_scores))

# Plot the performance of each iteration
for i in range(len(mean_test_scores)):
    plt.errorbar(i, -mean_test_scores[i], yerr=std_test_scores[i], fmt='o', capsize=5, color=colors(i))

plt.show()
