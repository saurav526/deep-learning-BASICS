# Encoder-Decoder Model using TensorFlow/Keras

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Parameters
latent_dim = 256
encoder_vocab_size = 1000
decoder_vocab_size = 1000

# ---------------------
# Encoder
# ---------------------
encoder_inputs = Input(shape=(None, encoder_vocab_size))

encoder_lstm = LSTM(latent_dim, return_state=True)

encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# Encoder states
encoder_states = [state_h, state_c]

# ---------------------
# Decoder
# ---------------------
decoder_inputs = Input(shape=(None, decoder_vocab_size))

decoder_lstm = LSTM(latent_dim,
                    return_sequences=True,
                    return_state=True)

decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs,
    initial_state=encoder_states
)

decoder_dense = Dense(decoder_vocab_size, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

# ---------------------
# Encoder-Decoder Model
# ---------------------
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()