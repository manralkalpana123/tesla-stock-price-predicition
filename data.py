import yfinance as yf
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)




def download_ticker(ticker: str = "TSLA", period: str = "10y") -> pd.DataFrame:
"""Download historical OHLCV data with yfinance."""
df = yf.download(ticker, period=period, auto_adjust=False)
df.index.name = 'Date'
return df




def save_raw(df: pd.DataFrame, name: str = "tsla_raw.csv") -> Path:
path = RAW_DIR / name
df.to_csv(path)
return path




if __name__ == "__main__":
df = download_ticker("TSLA", period="5y")
print(f"Downloaded {len(df)} rows")
save_raw(df)