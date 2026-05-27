import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(32, input_shape=(100, 1)),
    Dense(1)
])

model.compile(optimizer='adam',
              loss='mse')

model.summary()