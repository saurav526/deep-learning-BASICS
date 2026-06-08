class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.self_attn = SelfAttention(d_model)

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self, x):

        attn_output, _ = self.self_attn(x)

        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        return x
    
encoder = TransformerEncoderLayer(
d_model=128,
d_ff=512
)

x = tf.random.normal((2, 20, 128))

output = encoder(x)

print(output.shape)
