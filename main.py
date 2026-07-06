import pandas as pd
from src.data_prep import prep_data, clean_data
from src.model import train_model, tune_threshold, save_model
from sklearn.model_selection import train_test_split

# First, we create a dataframe from the heart_disease dataset
df = pd.read_csv('data/heart_disease.csv')
# Now we clean the data
clean_data(df)
# Now take the cleaned data and prep it for training (prep_data yields features, target, encoder, scaler in respectively)
cleaned = pd.read_csv('outputs/cleaned_data.csv')
X, y, con_cols, cat_cols = prep_data(cleaned, target='Heart Disease Status')
# Now, let's make the train/val/test split of the data
## First split -> create train and temp from X and y
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
## Second split -> create val and test from X_temp and y_temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Now that we have the data split up, we can begin training.
model = train_model(X_train, y_train, cat_cols, con_cols)
# We want to tune the decision threshold since we are more focused on reducing false negatives.
tuned = tune_threshold(model, X_val, y_val)
# Let's now go ahead and save the tuned model to our outputs
save_model(tuned, 'model')