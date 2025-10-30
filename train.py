import argparse
import numpy as np
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.models import build_lstm, build_simple_rnn


PROCESSED_DIR = Path(__file__).resolve().parents[1] / 'data' / 'processed'
MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)




def load_data():
X_train = np.load(PROCESSED_DIR / 'X_train.npy')
X_test = np.load(PROCESSED_DIR / 'X_test.npy')
y_train = np.load(PROCESSED_DIR / 'y_train.npy')
y_test = np.load(PROCESSED_DIR / 'y_test.npy')
return X_train, X_test, y_train, y_test




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['lstm', 'rnn'], default='lstm')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
args = parser.parse_args()


X_train, X_test, y_train, y_test = load_data()
# shape: (samples, seq_len, 1)


input_shape = (X_train.shape[1], X_train.shape[2])


if args.model == 'lstm':
model = build_lstm(input_shape)
model_name = MODELS_DIR / 'lstm_model.h5'
else:
model = build_simple_rnn(input_shape)
model_name = MODELS_DIR / 'rnn_model.h5'


checkpoint = ModelCheckpoint(str(model_name), save_best_only=True, monitor='val_loss')
early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)


history = model.fit(
X_train, y_train,
validation_data=(X_test, y_test),
epochs=args.epochs,
batch_size=args.batch,
callbacks=[checkpoint, early]
)


print('Training complete. Model saved to', model_name)