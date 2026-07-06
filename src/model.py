import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import precision_score, recall_score, fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_model(X_train, y_train, cat_cols, con_cols):
	"""
	Takes in the training data subset and returns a trained logistic regression model.

	Inputs:
	X_train - np.array():
		The input (training) feature data from which the model will derive weights.
	y_train - np.array():
		The (training) target feature.

	Output:
	model - LogisticRegression()
		The trained logistic regression model with the optimal combination of parameters.
	"""
	# define a ColumnTransformer instance to handle the preprocessing steps in the pipeline.
	preproc = ColumnTransformer(
		transformers=[
			('scale', StandardScaler(), con_cols),
			('encode', OneHotEncoder(drop='if_binary'), cat_cols)
		],
		remainder='drop'
	)
	# Now we want to create a Pipeline instance that will handle the preprocessing and fitting of the model.
	pipe = Pipeline([
		('preprocessing', preproc),
		('clf', LogisticRegression(max_iter=500, random_state=42))
	])
	# Define the range of parameters to be searched
	param_dist = {
		'clf__solver': ['lbfgs', 'liblinear'],
		'clf__C': np.logspace(-2, 3, 10),
	}
	# Create GridSearchCV object that will search param_dist for the optimal parameters
	gs = GridSearchCV(
		pipe,
		param_dist,
		scoring=make_scorer(fbeta_score, beta=2)
	)
	# Fit gridsearch to training data
	gs.fit(X_train, y_train)
	# Return the best performing model from the gridsearch
	model = gs.best_estimator_
	return model


def tune_threshold(model, X_val, y_val):
	"""
	Takes in the optimal fitted model and, using a distinct split of the data, tunes the decision threshold.
	This will help the model prioritize minimizing false negatives when testing is conducted, and will
	store that prioritization for later inference.

	Inputs:
	model - trained LogisticRegression():
		The trained model who's decision threshold will be adjusted.
	X_val - np.array():
		the validation portion of the features.
	y_val - np.array():
		the validation portion of the target feature.

	Ouptut:
	tuned - TunedThresholdClassifierCV():
		a fitted TunedThresholdClassifierCV() object which contains the tuned decition threshold, as well as the model weights.
	"""
	# Create a TunedThresholdClassifierCV object, which takes in the trained model, and uses F2 scoring to evalutate
	tuned = TunedThresholdClassifierCV(
		estimator=model,
		scoring=make_scorer(fbeta_score, beta=2), # type: ignore
		cv='prefit', # Since the model has already been fitted, we do not want threshold classifier to re-run cross-validation
		refit=False,
		random_state=42
	)
	# Fit to the validation data to determine the tuned threshold.
	tuned.fit(X_val, y_val)
	# Return the fitted TunedThreshold
	return tuned

def model_performance(model, X_test, y_test):
	"""
	Takes in the trained model and the testing data and returns the precision, recall, and fbeta scores.

	Inputs:
	model - Trained LogisticRegression() model:
		The trained model
	test_data - pd.DataFrame:
		The testing split of the dataset 

	Outputs:
	p - float:
		the model's precision score
	r - float:
		the model's recall score
	fbeta - float:
		the model's fbeta score
	"""
	# Use model to predict target values for X_test
	preds = model.predict(X_test)
	# Find the recall, precision, and fbeta score
	r = recall_score(y_test, preds)
	p = precision_score(y_test, preds)
	fbeta = fbeta_score(y_test, preds, beta=2)
	# Return r, p, and fbeta
	return p, r, fbeta
	
def save_model(model, path):
	"""
	Takes the model and saves it as a serialized .pkl file in the project's 'outputs' folder.

	Input:
	model - Trained LogisticRegression():
		the trained model
	
	Output:
	NA (model will appear in 'outputs')
	"""
	# Create context window for a new file where the serialized model will be saved.
	with open(f'outputs/{path}.pkl', 'wb') as f:
		pkl.dump(model, f)

def load_model(path):
	"""
	Takes in the filepath for the serialized model and returns the trained model as a callable object.

	Input:
	filepath - str:
		the relative filepath to the serialized model.
	
	Output:
	model - LogisticRegression() Obj:
		callable model.
	"""
	# Fetch the saved model at the given path and return it as a callable object.
	with open(f'outputs/{path}.pkl', 'rb') as f:
		model = pkl.load(f)
	# Return the model
	return model