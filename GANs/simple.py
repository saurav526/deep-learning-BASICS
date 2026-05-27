import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generator
generator = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(784, activation='sigmoid')
])

# Discriminator
discriminator = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(1, activation='sigmoid')
])

print("Generator")
generator.summary()

print("Discriminator")
discriminator.summary()