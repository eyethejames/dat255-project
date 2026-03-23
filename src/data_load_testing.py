from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/m5-forecasting-accuracy")

calendar = pd.read_csv(DATA_DIR / "calendar.csv")
sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

## bekrefte at filene leses inn, kolonnene ser riktige ut og datastrukturene er som forventet
# print("calendar:", calendar.shape)
# print("sales:", sales.shape)
# print("prices:", prices.shape)

# print("\nSales columns:", sales.columns)
# print(sales.columns[:15])

# print("\nCalendar columns:", calendar.columns)
# print(calendar.columns[:15])

# print("\nPrices columns:", prices.columns)
# print(prices.columns[:15])

# print("\nSamples sales rows:")
# print(sales.head())

# skriver ut unike verdier for butikker, kategorier og avdelinger
print("\nStores:", sales["store_id"].unique())
print("\nCategories:", sales["cat_id"].unique())
print("\nDepartments:", sales["dept_id"].unique())

# filtrere salg for en spesifikk butikk og kategori
subset = sales [
    (sales["store_id"] == "CA_1") &
    (sales["cat_id"] == "FOODS") & # redundant since dept_id is more specific, but just to be sure
    (sales["dept_id"] == "FOODS_1")
    ].copy()

print(subset.shape)
print(subset[["id", "item_id", "dept_id", "cat_id", "store_id"]].head())

# henter ut dag-kolonnene
day_cols = [col for col in subset.columns if col.startswith("d_")]
print("\nDay columns:", len(day_cols))
print("First 5 day columns:", day_cols[:5])
print("Last 5 day columns:", day_cols[-5:])

# sjekk en faktisk tidsserie
sample_series = subset.iloc[0][day_cols].astype(int)  # og konverterer til int for enklere analyse
print(sample_series.head())
print(sample_series.tail())

# etablerer sliding windows for å lage treningsdata
def create_windows(series, input_window=28, target_window=7):
    X, y = [], []

    for i in range(len(series) - input_window - target_window + 1):
        x_window = (series[i:i+input_window].values)
        y_window = (series[i+input_window:i+input_window+target_window].values)

        X.append(x_window)
        y.append(y_window)

    return X, y

X, y = create_windows(sample_series)

# viser antall windows og et eksempel på første vindu
print("\nAntall X-vinduer:", len(X))
print("Antall y-vinduer:", len(y))
print("Første X-vindu:", X[0])
print("Første y-vindu:", y[0])

# sjekk at vinduene er forskjøvet riktig
print(X[1], y[1])
print(X[2], y[2])
print(X[3], y[3])