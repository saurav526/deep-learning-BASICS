import tensorflow as tf
from tensorflow.keras import layers
class SelfAttention(layers.Layer):
    def __init__(self, d_model):
        super().__init__()

        self.Wq = layers.Dense(d_model)
        self.Wk = layers.Dense(d_model)
        self.Wv = layers.Dense(d_model)

    def call(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)

        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights

x = tf.random.normal((2, 5, 128))

attn = SelfAttention(128)

output, weights = attn(x)

print(output.shape)
# (2, 5, 128)tion_weights

