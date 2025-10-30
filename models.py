from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout




def build_simple_rnn(input_shape, units=50):
model = Sequential()
model.add(SimpleRNN(units, return_sequences=False, input_shape=input_shape))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
return model




def build_lstm(input_shape, units=50):
model = Sequential()
model.add(LSTM(units, return_sequences=False, input_shape=input_shape))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
return model