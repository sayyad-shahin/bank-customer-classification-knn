import pytest

from common import load_dataset
from predict import MODEL_PATH, load_bundle
from train import train


@pytest.fixture(scope="session")
def bundle():
    """Use the shipped model (tests what users get); retrain only if it is missing or incompatible."""
    try:
        return load_bundle(MODEL_PATH)
    except (FileNotFoundError, RuntimeError):
        X, y, ids = load_dataset("data/bank_customers.csv")
        return train(X, y, ids)[0]
