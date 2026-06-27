import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prep_data(X, cat_vars=[], target=None, encoder=None, scaler=None, for_training=True):
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
	# Separate the target column from the other columns
	if target is not None:
		y = X[target]
		X = X.drop([target], axis=1)
	else:
		y = np.array([])
	
	# Separate the categorical and continuous columns
	X_cat = X[[cat_vars]]
	con_cols = X.select_dtypes(include=['numbers']).columns.tolist()
	X_con = X[con_cols]

	# When prepping to train model
	if for_training == True:
		encoder = OneHotEncoder()
		scaler = StandardScaler()
		X_cat = encoder.fit_transform(X_cat)
		X_con = scaler.fit_transform(X_con)
	# When processing input for inference
	else:
		X_cat = encoder.fit_transform(X_cat)
		X_con = scaler.fit_transform(X_con)
	# Combine scaled/encoded features and return output
	X = np.concatenate([X_cat, X_con], axis=1)
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
