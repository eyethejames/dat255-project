from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/m5-forecasting-accuracy")

calendar = pd.read_csv(DATA_DIR / "calendar.csv")
sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
prices = pd.read_csv(DATA_DIR / "sell_prices.csv")