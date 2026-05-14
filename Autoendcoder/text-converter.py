import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ( Input,Embedding,LSTM,Dense)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

sentences = [
    "i love deep learning",
    "deep learning is powerful",
    "i love artificial intelligence",
    "machine learning is amazing",
    "artificial intelligence is future"
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)
input_sequences = []
target_words = []

for seq in sequences:

    for i in range(1, len(seq)):

        # Input Sequence
        input_seq = seq[:i]

        # Target Word
        target = seq[i]

        input_sequences.append(input_seq)
        target_words.append(target)

max_len = max(len(seq) for seq in input_sequences)

X = pad_sequences(
    input_sequences,
    maxlen=max_len,
    padding='pre'
)
y = to_categorical(
    target_words,
    num_classes=vocab_size
)

print("Input Shape:", X.shape)
print("Output Shape:", y.shape)

inputs = Input(shape=(max_len,))

embedding = Embedding(
    input_dim=vocab_size,
    output_dim=64,
    input_length=max_len
)(inputs)

encoder = LSTM(
    128,
    return_sequences=False
)(embedding)

decoder = Dense(
    64,
    activation='relu'
)(encoder)

outputs = Dense(
    vocab_size,
    activation='softmax'
)(decoder)

model = Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

def generate_text(seed_text, next_words):

    for _ in range(next_words):

        # Convert Text to Sequence
        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        # Pad Sequence
        token_list = pad_sequences(
            [token_list],
            maxlen=max_len,
            padding='pre'
        )

        # Predict Probabilities
        predicted_probs = model.predict(
            token_list,
            verbose=0
        )

        # Get Highest Probability Word
        predicted_index = np.argmax(predicted_probs)

        # Convert Index to Word
        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted_index:
                output_word = word
                break

        # Append Word
        seed_text += " " + output_word

    return seed_text
print(
    generate_text(
        "i love",
        3
    )
)