import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# Sample input
x = tf.random.normal((1, 10, 64))

# Multi-head attention
attention = MultiHeadAttention(num_heads=4, key_dim=64)

output = attention(x, x)

# Normalize
output = LayerNormalization()(output)

print(output.shape)