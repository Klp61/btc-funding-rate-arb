# scripts/run_pipeline.py

from src.dataset import build_dataset

if __name__ == "__main__":
    df = build_dataset("BTCUSDT")
    print(df)