import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prep_data(X, target=None, encoder=None, scaler=None, for_training=True):
	"""
	Takes in an dataframe containing the data to be used in the analysis and prepares it for either inference
	or training.

	Inputs:
	X - DataFrame:
		The data needing prepped
	cat_vars - list:
		A list of column names for the categorical variables in X.
	target - str:
		The name of the target column in the dataset.
	encoder - sklearn.preprocessing.OneHotEncoder:
		Trained encoder to be used to encode categorical variables.
	scaler - sklearn.preprocessing.StandardScaler:
		Trained scaler to be used to sccale continuous variables.
	for_training - bool:
		Indicates if the data is being prepped for training (True), or will be used for inference (False).

	Outputs:
	X
	y
	encoder
	scaler
	"""
	# Separate the target column from the other columns (if no target column, assign y to empty array.)
	if target is not None:
		y = X[target]
		X = X.drop([target], axis=1)
	else:
		y = np.array([])
	
	# Separate the categorical, continuous and binary columns
	X_cat = X.loc[:, X.nunique() == 3]
	con_cols = X.select_dtypes(include=['numbers']).columns.tolist()
	X_con = X[con_cols]
	X_bin = X.loc[:, X.nunique() == 2]

	# Fit/transform column groups with the necessary transformer (simple .replace() for binary columns)
	if for_training == True:
		encoder = OneHotEncoder()
		scaler = StandardScaler()
		X_cat = encoder.fit_transform(X_cat)
		X_con = scaler.fit_transform(X_con)
		X_bin = X_bin.replace({'Yes': 1, 'No': 0})
		# We also need to convert target values to numeric from string.
		y = y.replace({'Yes': 1, 'No': 0})
	# When processing input for inference
	else:
		X_cat = encoder.fit_transform(X_cat)
		X_con = scaler.fit_transform(X_con)
		X_bin = X_bin.replace({'Yes': 1, 'No': 0})
		try:
			y = y.replace({'Yes': 1, 'No': 0})
		except AttributeError:
			pass
	# Combine scaled/encoded features and return output
	X = np.concatenate([X_cat, X_con, X_bin], axis=1)
	return X, y, encoder, scaler

def clean_data(data):
	"""
	Performs cleaning on the passed dataset.

	Input:
	data - pd.DataFrame:
		The dataframe to be cleaned
	
	Output:
	cleaned_data - pd.DataFrame:
		The cleaned dataframe
	"""
	# Separate the column(s) that need imputing instaed of dropping nans.
	impute_cols = data.loc[:, data.isna().mean() > 0.05].index
	drop_cols = data.loc[:, data.isna().mean() <= 0.05].index
	# An array of the remaining row indices present after na's are dropped from drop_cols.
	# This will ensure that we keep the same original indices when imputed columns and dropna columns
	# are recombined in the final cleaned df.
	master_row_ind = data[drop_cols].dropna().index
	# Use row index to ensure the same original df row indices are kept between both groups.
	impute = data.loc[master_row_ind, impute_cols]
	dropped = data.loc[master_row_ind, drop_cols]
	# Now we impute the missing values in 'impute'
	imputed = impute.fillna('Missing')
	# Now that we have columns imputed and leftover na's dropped, we recombine them into the cleaned df
	cleaned_df = pd.concat([dropped, imputed], axis=1).reset_index(drop=True)
	# Final check to make sure there are no remaining na's
	assert cleaned_df.isna().sum().sum() == 0
	# Save the cleaned data as a csv file in the outputs/ directory
	cleaned_df.to_csv('outputs/cleaned_data.csv', index=False)