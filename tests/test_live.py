import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

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


render_url = "https://ml-model-deployment-lzxm.onrender.com/predict"

request = requests.post(render_url, json=sample)
assert request.status_code == 200

logging.info(f"Status code: {request.status_code}")
logging.info(f"Response body: {request.json()}")
