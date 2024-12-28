import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load the VIF dataset
vif_data = pd.read_excel('data_preprocessing.xlsx')

# Prepare the data
X_vif = vif_data[['스트레스인지율', '우울감경험율', '스트레스로인한정신상담률', '우울증상유병률', '자살생각율', '주관적건강인지율']]

# Adding constant to the model for intercept term
X_vif = add_constant(X_vif)

# Calculate VIF for each feature
vif = pd.DataFrame()
vif['Variable'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

vif.to_excel('vif_output.xlsx', engine='openpyxl', index=False)

