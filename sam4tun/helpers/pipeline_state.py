import os
import pickle


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pipeline state: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)
