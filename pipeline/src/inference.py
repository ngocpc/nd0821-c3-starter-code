"""
Script to evaluate model performance on slices of the data for categorical features.
"""

import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.model import compute_model_metrics
from ml.data import (
    process_data,
    get_cat_features
)

import logging
logger = logging.getLogger(__name__)


data_path = 'data/census.csv'
model_path = 'pipeline/model'

def model_performance():
    """ output model performance on categorical features """

    data = pd.read_csv(data_path)
    _, test = train_test_split(data, test_size=0.20)

    model = joblib.load(os.path.join(model_path, 'model.pkl'))
    encoder = joblib.load(os.path.join(model_path, 'encoder.pkl'))
    lb = joblib.load(os.path.join(model_path, 'lb.pkl'))

    model_metrics = []
    categorical_features = get_cat_features()

    for cat_feature in categorical_features:
        for cls in test[cat_feature].unique():
            df_sub = test[test[cat_feature] == cls]

            X_test, y_test, _, _ = process_data(
                df_sub,
                categorical_features,
                label='salary',
                encoder=encoder,
                lb=lb,
                training=False
            )

            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f'{cat_feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}'
            model_metrics.append(row)

            with open('pipeline/results/model_metrics.txt', 'w') as file:
                for row in model_metrics:
                    file.write(row + '\n')

if __name__ == '__main__':
    model_performance()
