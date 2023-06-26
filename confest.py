import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.src.ml.data import process_data
from pipeline.src.ml.model import train_model
from fastapi.testclient import TestClient
from pipeline.main import app

@pytest.fixture()
def client():
    client = TestClient(app)
    return client
