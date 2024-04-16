import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Dropout
from layers import *

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_encoder_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerClassifier, self).__init__()
        # Lập trình tại đây
        self.d_model=d_model
        self.embedding =  Embedding(input_dim=input_vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_encoder_layers)]
        self.dropout = Dropout(rate)
        self.global_average_pooling = GlobalAveragePooling1D()
        self.final_layer = Dense(1,activation='sigmoid')

    def call(self, x, training):
        # Lập trình tại đây
        q=x
        embedded=self.embedding(q)
        embedded *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos=positional_encoding(q.shape[1],self.d_model)
        encoder_out = self.dropout(embedded + pos, training=training)
        for encoder_layer in self.enc_layers:
            encoder_out = encoder_layer(encoder_out, training, mask=None)
        pooled=self.global_average_pooling(encoder_out)
        output=self.final_layer(pooled)
        return output