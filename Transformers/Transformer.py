import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# =====================================================
# Positional Encoding
# =====================================================

def positional_encoding(position, d_model):

    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000,
        (2 * (np.arange(d_model)[np.newaxis, :] // 2))
        / np.float32(d_model)
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)



class MultiHeadAttention(layers.Layer):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        x = tf.reshape(
            x,
            (batch_size, -1,
             self.num_heads,
             self.depth)
        )

        return tf.transpose(x,
                            perm=[0,2,1,3])

    def scaled_dot_product_attention(
            self,
            q,
            k,
            v,
            mask=None):

        matmul_qk = tf.matmul(
            q,
            k,
            transpose_b=True
        )

        dk = tf.cast(
            tf.shape(k)[-1],
            tf.float32
        )

        scaled_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(
            scaled_logits,
            axis=-1
        )

        output = tf.matmul(
            attention_weights,
            v
        )

        return output, attention_weights

    def call(self,
             q,
             k,
             v,
             mask=None):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention, weights = \
            self.scaled_dot_product_attention(
                q, k, v, mask
            )

        attention = tf.transpose(
            attention,
            perm=[0,2,1,3]
        )

        concat_attention = tf.reshape(
            attention,
            (batch_size,
             -1,
             self.d_model)
        )

        output = self.dense(
            concat_attention
        )

        return output


# =====================================================
# Feed Forward
# =====================================================

def point_wise_ffn(d_model, dff):

    return tf.keras.Sequential([
        layers.Dense(dff,
                     activation="relu"),
        layers.Dense(d_model)
    ])



class EncoderLayer(layers.Layer):

    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 rate=0.1):

        super().__init__()

        self.mha = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ffn = point_wise_ffn(
            d_model,
            dff
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):

        attn = self.mha(
            x, x, x
        )

        attn = self.dropout1(
            attn,
            training=training
        )

        x = self.norm1(
            x + attn
        )

        ffn_output = self.ffn(x)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        x = self.norm2(
            x + ffn_output
        )

        return x


# =====================================================
# Decoder Layer
# =====================================================

class DecoderLayer(layers.Layer):

    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 rate=0.1):

        super().__init__()

        self.masked_mha = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.cross_mha = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ffn = point_wise_ffn(
            d_model,
            dff
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm3 = layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self,
             x,
             enc_output,
             look_ahead_mask=None):

        # Masked Self Attention
        attn1 = self.masked_mha(
            x,
            x,
            x,
            look_ahead_mask
        )

        x = self.norm1(
            x + attn1
        )

        # Cross Attention
        attn2 = self.cross_mha(
            x,
            enc_output,
            enc_output
        )

        x = self.norm2(
            x + attn2
        )

        ffn_output = self.ffn(x)

        x = self.norm3(
            x + ffn_output
        )

        return x


# =====================================================
# Encoder
# =====================================================

class Encoder(layers.Layer):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 vocab_size,
                 maximum_position_encoding):

        super().__init__()

        self.d_model = d_model

        self.embedding = layers.Embedding(
            vocab_size,
            d_model
        )

        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )

        self.enc_layers = [
            EncoderLayer(
                d_model,
                num_heads,
                dff
            )
            for _ in range(num_layers)
        ]

    def call(self, x):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(
                self.d_model,
                tf.float32
            )
        )

        x += self.pos_encoding[
            :,
            :seq_len,
            :
        ]

        for layer in self.enc_layers:
            x = layer(x)

        return x


# =====================================================
# Decoder
# =====================================================

class Decoder(layers.Layer):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 vocab_size,
                 maximum_position_encoding):

        super().__init__()

        self.d_model = d_model

        self.embedding = layers.Embedding(
            vocab_size,
            d_model
        )

        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )

        self.dec_layers = [
            DecoderLayer(
                d_model,
                num_heads,
                dff
            )
            for _ in range(num_layers)
        ]

    def call(self,
             x,
             enc_output,
             look_ahead_mask=None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(
                self.d_model,
                tf.float32
            )
        )

        x += self.pos_encoding[
            :,
            :seq_len,
            :
        ]

        for layer in self.dec_layers:
            x = layer(
                x,
                enc_output,
                look_ahead_mask
            )

        return x


# =====================================================
# Transformer
# =====================================================

class Transformer(tf.keras.Model):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 max_pos=1000):

        super().__init__()

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            max_pos
        )

        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            max_pos
        )

        self.final_layer = layers.Dense(
            target_vocab_size
        )

    def call(self,
             inp,
             tar,
             look_ahead_mask=None):

        enc_output = self.encoder(inp)

        dec_output = self.decoder(
            tar,
            enc_output,
            look_ahead_mask
        )

        final_output = self.final_layer(
            dec_output
        )

        return final_output