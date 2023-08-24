from pipeline.main import app
from fastapi.testclient import TestClient

import logging
logger = logging.getLogger(__name__)

client = TestClient(app)

logger.info(f"client: {client}")

def test_get():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'greeting': 'Hello World!'}

def test_local_api_predict():

    sample = {
        'age': 45,
        'workclass': 'State-gov',
        'fnlgt': 65241,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Sales',
        'relationship': 'Wife',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 1949,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States',
    }

    response = client.post('/predict', json=sample)
    logger.info(f'response: {response}')

    assert response.status_code == 200
    assert response.json()['prediction'] == '<=50K'
