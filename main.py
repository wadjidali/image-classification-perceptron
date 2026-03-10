from src.dataset import load_dataset
from src.model import train_model

X, y = load_dataset("data/raw")

model = train_model(X, y)

print("Model trained successfully")
