import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)




def load_raw(path: Path) -> pd.DataFrame:
df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
return df




def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
# Keep needed columns and forward-fill missing
df = df.copy()
df = df[['Open','High','Low','Close','Adj Close','Volume']]
df = df.sort_index()
df = df.fillna(method='ffill').dropna()
return df




def scale_series(series: pd.Series, scaler=None):
arr = series.values.reshape(-1, 1)
if scaler is None:
scaler = MinMaxScaler(feature_range=(0, 1))
arr_scaled = scaler.fit_transform(arr)
else:
arr_scaled = scaler.transform(arr)
return arr_scaled, scaler




def create_sequences(values: np.ndarray, seq_len: int = 60):
X, y = [], []
for i in range(seq_len, len(values)):
X.append(values[i - seq_len:i])
y.append(values[i])
X = np.array(X)
y = np.array(y)
return X, y




def prepare_data(tsla_csv: str, seq_len: int = 60, test_split: float = 0.2):
df = load_raw(Path(tsla_csv))
df = basic_cleaning(df)


# Use 'Close' price for prediction
close_scaled, scaler = scale_series(df['Close'])


X, y = create_sequences(close_scaled, seq_len=seq_len)


# train/test split
n_test = int(len(X) * test_split)
X_train, X_test = X[:-n_test], X[-n_test:]
y_train, y_test = y[:-n_test], y[-n_test:]


# Save processed
np.save(PROCESSED_DIR / 'X_train.npy', X_train)
np.save(PROCESSED_DIR / 'X_test.npy', X_test)
np.save(PROCESSED_DIR / 'y_train.npy', y_train)
np.save(PROCESSED_DIR / 'y_test.npy', y_test)


# Save scaler for inverse-transform
import joblib
joblib.dump(scaler, PROCESSED_DIR / 'scaler.save')


return X_train, X_test, y_train, y_test, scaler




if __name__ == "__main__":
prepare_data(str(RAW_DIR / 'tsla_raw.csv'))