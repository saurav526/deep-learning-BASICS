import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Encoder
input_layer = Input(shape=(20,))
encoded = Dense(10, activation='relu')(input_layer)

# Decoder
decoded = Dense(20, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy')

autoencoder.summary()