import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import mnist

# Load MNIST Dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten Images
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# Encoder Dimension
encoding_dim = 32

# Input Layer
input_img = Input(shape=(784,))

# Encoder
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded_output = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded_output)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(784, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = Model(input_img, decoded_output)

# Encoder Model
encoder = Model(input_img, encoded_output)

# Compile Model
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train Model
autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Encode and Decode Images
encoded_imgs = encoder.predict(x_test)

decoded_imgs = autoencoder.predict(x_test)

# Display Results
n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # Original Image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Image
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()