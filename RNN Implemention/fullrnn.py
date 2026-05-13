import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Sample Data
data = np.array([i for i in range(100)]).reshape(-1, 1)

# Normalize Data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Sequence Length
n_input = 5

# Generate Sequences
generator = TimeseriesGenerator(data, data, length=n_input, batch_size=1)

# Build RNN Model
model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(n_input, 1)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model
model.fit(generator, epochs=50)

# Prediction
test_input = data[-n_input:]
test_input = test_input.reshape((1, n_input, 1))

predictions = []

for i in range(20):
    pred = model.predict(test_input, verbose=0)
    predictions.append(pred[0][0])

    test_input = np.append(test_input[:, 1:, :], [[pred]], axis=1)

# Convert Back to Original Scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plot Results
plt.plot(range(100), scaler.inverse_transform(data), label='Original Data')
plt.plot(range(100, 120), predictions, label='Predictions')
plt.legend()
plt.show()