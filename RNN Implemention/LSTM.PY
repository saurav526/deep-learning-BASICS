import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Create Sample Data
data = np.array([i for i in range(200)]).reshape(-1, 1)

# Normalize Data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Number of Previous Time Steps
n_input = 10

# Create Time-Series Generator
generator = TimeseriesGenerator(
    data,
    data,
    length=n_input,
    batch_size=1
)


model = Sequential([
    
    # LSTM Layer
    LSTM(
        64,
        activation='tanh',
        input_shape=(n_input, 1)
    ),

    # Hidden Dense Layer
    Dense(32, activation='relu'),

    # Output Layer
    Dense(1)
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train Model
model.fit(
    generator,
    epochs=50,
    verbose=1
)

# Take Last Sequence for Prediction
test_input = data[-n_input:]

# Reshape into 3D
test_input = test_input.reshape((1, n_input, 1))

# Store Predictions
predictions = []

# Predict Next 20 Values
for i in range(20):

    # Predict
    pred = model.predict(test_input, verbose=0)

    # Save Prediction
    predictions.append(pred[0][0])

    # Update Input Sequence
    test_input = np.append(
        test_input[:, 1:, :],
        [[pred]],
        axis=1
    )

# Convert Predictions Back to Original Scale
predictions = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

# Plot Original Data
plt.plot(
    range(200),
    scaler.inverse_transform(data),
    label='Original Data'
)

# Plot Predictions
plt.plot(
    range(200, 220),
    predictions,
    label='Predictions'
)

# Show Labels
plt.legend()

# Display Graph
plt.show()
