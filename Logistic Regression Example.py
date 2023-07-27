# -*- coding: utf-8 -*-
"""
An example on Logistic Regression
Used with the permission of other participants from the Virginia Datathon 2023 team
"""
# Import sodapy and other packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
import sodapy as spy
from sklearn.metrics import classification_report, confusion_matrix
# Create Socrata client
data_url = "data.virginia.gov"
data_set = "b6ux-y6yi"
client = spy.Socrata(data_url, app_token= "wEiZLz2RwQ5niCoBwXyFIptzm",
                     timeout=50)

# The line below gives us a list of dictionaries
results = client.get(data_set)
# Convert the list of dictionaries into a dataframe
maternity_data = pd.DataFrame.from_records(results)
# See the shape of the dataframe to explore how many rows and columns we have
maternity_data.shape
# See unique values of our target
maternity_data["gestational_age"].unique()
# Recode the variable to at risk if it is before Early Term
# This is based on a consultation with a subject matter expert
# 0 is low risk, while 1 is high risk. Therefore, this is a classification problem
maternity_data["gestional_age_risk"] = np.select([maternity_data["gestational_age"].isin(['Early Term', 'Full Term', 'Late Term']),
                                                    maternity_data["gestational_age"].isin(['Late Preterm', 'Very Preterm',
       'Extremely Preterm', 'Post Term', 'Moderate Preterm'])]
                                                    , [0, 1])
# Subset the dataframe to the variables that we will use
maternity_subset = maternity_data[["medicaid_program", "delivery_system", "mco_count", "cont_enroll_category",
                                   "gravidity", "maternal_raceeth", "maternal_age", "maternal_asthma",
                                   "maternal_diabetes", "gestational_diabetes", "pnc_index", "birth_weight",
                                   "amb_utilization", "prenatal_depscr", "gestional_age_risk"]]
# Check that the dataset does not contain any nulls
maternity_subset.isna().sum()
# Recode logical columns from False and True to 0 and 1
maternity_subset['amb_utilization'] = maternity_subset['amb_utilization'] * 1
maternity_subset['prenatal_depscr'] = maternity_subset['prenatal_depscr'] * 1
# Subset the target and the predictors
# Notice that all predictors are categorical
predictors = maternity_subset.loc[:, maternity_subset.columns!="gestional_age_risk"]
target = maternity_subset["gestional_age_risk"]

# Get dummies from predictors, all variables are categorical
predictors_w_dummies = pd.get_dummies(predictors)

# Create the first model
# The random state is required to allow to reproduce the results
log_reg = LogisticRegression(random_state = 450)
# Fit the model
log_reg.fit(predictors_w_dummies, target)
# See the classes
log_reg.classes_

# Get the formula from the model
log_reg.intercept_
log_reg.coef_

# See how accurate the model is with 10-fold cross-validation
cross_val_score(log_reg, predictors_w_dummies, target, cv = 10).mean()

# Get the coefficients of the model, as we are mostly interested on them
coef_dict = {}
for coef, feature in zip(log_reg.coef_[0, :], predictors_w_dummies.columns):
    coef_dict[feature] = coef
# Make it a dataframe
coef_df = pd.DataFrame.from_dict(coef_dict.items())
# Calculate the standard deviation of each predictor
coef_df["Standard_Deviation"] = np.std(predictors_w_dummies, 0).tolist()
coef_df.sort_values(by = [1], ascending = False, inplace = True)
coef_df["Importance"] = coef_df["Standard_Deviation"] * coef_df[1]
# Filter importances that are less than 0.25 or more than -0.25
# This approach is used as sklearn lacks hypothesis testing for coefficients with p-values
coef_filtered = coef_df[(coef_df["Importance"] > 0.15) | (coef_df["Importance"] < -0.15)]

# Get the name of the selected variables
filtered_predictors = predictors_w_dummies[coef_filtered[0].tolist()]
# Train a new model only using the new variables
refined_log_reg = LogisticRegression(random_state = 500)
refined_log_reg.fit(filtered_predictors, target)

# Get the cross-validated score
cross_val_score(refined_log_reg, filtered_predictors, target, cv = 10).mean()

# Get the coefficients of the second model
coef_dict_2 = {}
for coef, feature in zip(refined_log_reg.coef_[0, :], filtered_predictors.columns):
    coef_dict_2[feature] = coef
# Make it a dataframe
coef_df_2 = pd.DataFrame.from_dict(coef_dict_2.items())
coef_df_2.sort_values(by = [1], ascending = False, inplace = True)
# Plot the coefficients 
sns.barplot(data = coef_df_2, x = 1, y = 0)
plt.xlabel("Coefficient value (Log odds)")
plt.title("Variables related to early pregnancies")
plt.show()