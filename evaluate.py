import argparse
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


PROCESSED_DIR = Path(__file__).resolve().parents[1] / 'data' / 'processed'
REPORTS_DIR = Path(__file__).resolve().parents[1] / 'reports' / 'figures'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)




def load_processed():
X_test = np.load(PROCESSED_DIR / 'X_test.npy')
y_test = np.load(PROCESSED_DIR / 'y_test.npy')
scaler = joblib.load(PROCESSED_DIR / 'scaler.save')
return X_test, y_test, scaler




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='path to model .h5')
args = parser.parse_args()


X_test, y_test, scaler = load_processed()
model = load_model(args.model)


preds = model.predict(X_test)


# inverse transform
preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
y_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


# simple plot
plt.figure(figsize=(10, 6))
plt.plot(y_inv, label='True')
plt.plot(preds_inv, label='Predicted')
plt.legend()
plt.title('Model predictions vs True')
out = REPORTS_DIR / (Path(args.model).stem + '_pred_vs_true.png')
plt.savefig(out)
print('Saved plot to', out)
