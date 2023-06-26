# Script to train machine learning model.
import pandas as pd
import logging
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from ml.data import process_data, get_cat_features
from ml.model import train_model, compute_model_metrics, inference


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logging.info("Importing data")

# Load in the data.
print(os.getcwd())
data_path = Path('data/census.csv')
data = pd.read_csv(data_path)

logging.info("Splitting data")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = get_cat_features()

# Process train data
logging.info("Process train data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logging.info("Process test data")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
# Train model
logging.info("Training model")
model = train_model(X_train, y_train)

# Evaluate model
logging.info("Evaluating the model on the test set")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save model
logging.info("Saving model")
joblib.dump(model, 'pipeline/model/model.pkl')
joblib.dump(encoder, 'pipeline/model/encoder.pkl')
joblib.dump(lb, 'pipeline/model/lb.pkl')
