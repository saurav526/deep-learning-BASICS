class CrossAttention(layers.Layer):
    def __init__(self, d_model):
        super().__init__()

        self.Wq = layers.Dense(d_model)
        self.Wk = layers.Dense(d_model)
        self.Wv = layers.Dense(d_model)

    def call(self, decoder_input, encoder_output):

        Q = self.Wq(decoder_input)

        K = self.Wk(encoder_output)

        V = self.Wv(encoder_output)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights
    
encoder_output = tf.random.normal((2, 10, 128))
decoder_input = tf.random.normal((2, 6, 128))

cross_attn = CrossAttention(128)

output, weights = cross_attn(
    decoder_input,
    encoder_output
)

print(output.shape)
