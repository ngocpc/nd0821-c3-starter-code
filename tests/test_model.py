"""
This script includes unit tests for the ML model

"""
import pytest
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pipeline.src.ml.data import process_data

import logging
logger = logging.getLogger(__name__)


data_path = 'data/census.csv'
model_path = 'pipeline/model/model.pkl'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='data')
def data():
    yield pd.read_csv(data_path)


def test_load_data(data):

    """ Check the data received """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_model():

    """ Test the model type """

    model = joblib.load(model_path)
    assert isinstance(model, RandomForestClassifier)


def test_process_data(data):

    """ Test the data split """

    train, _ = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(train, cat_features, label='salary')
    assert len(X) == len(y)
