import tensorflow as tf
from tensorflow.keras import layers, models

class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.ff_dim = ff_dim

        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        return x

inputs = tf.keras.Input(shape=(25, 1280))
x = TransformerBlock(dim=1280, heads=4, ff_dim=512)(inputs)
model = tf.keras.Model(inputs, x)

print('Keras 3 Layer Names:')
for layer in model.layers[1].weights:
    print(layer.name, layer.shape)

