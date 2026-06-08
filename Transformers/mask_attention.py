class MaskedSelfAttention(layers.Layer):
    def __init__(self, d_model):
        super().__init__()

        self.Wq = layers.Dense(d_model)
        self.Wk = layers.Dense(d_model)
        self.Wv = layers.Dense(d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(dk)

        # Upper triangular mask
        mask = 1 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)),
            -1,
            0
        )

        scores += mask * (-1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights