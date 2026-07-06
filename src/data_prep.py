import pandas as pd

def prep_data(X, target):
	"""
	Takes in an dataframe containing the data to be used in the analysis and prepares it for either inference
	or training.

	Inputs:
	X - DataFrame:
		The data needing prepped
	target - str:
		The name of the target column in the dataset.


	Outputs:
	X - pd.DataFrame:
		Dataframe contianing the data features
	y - pd.Series:
		Pandas series containing the target column.
	con_cols - list:
		A list containing the column names for the continuous features
	cat_cols - list:
		A list containing the column names for the categorical features
	"""
	# Separate the target column from the other columns (if no target column, assign y to empty array.)
	y = X[target]
	X = X.drop([target], axis=1)
	
	# Separate the categorical and continuous columns
	con_cols = X.select_dtypes(include=['number']).columns.tolist()
	cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
	# Transform y to numeric values instead of
	y = y.replace({'Yes': 1, 'No': 0})

	return X, y, con_cols, cat_cols

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
	impute_cols = data.loc[:, data.isna().mean() > 0.05].columns
	drop_cols = data.loc[:, data.isna().mean() <= 0.05].columns
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
	# Return the cleaned dataframe.
	return cleaned_df