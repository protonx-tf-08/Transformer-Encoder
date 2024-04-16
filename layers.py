import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

def scaled_dot_product_attention(q, k, v, mask):
    # Lập trình tại đây
    d_k = tf.cast(tf.shape(q)[-1],dtype=tf.float32)
    a = tf.matmul(q,k,transpose_b=True)/tf.math.sqrt(d_k)
    if mask!=None:
        a-=mask*float('inf')
    a=tf.nn.softmax(a)
    output = tf.matmul(a,v)
    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = int(d_model/num_heads)
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        # Lâp trình tại đây
        length=tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, length, self.num_heads, self.depth)) 
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def call(self, v, k, q, mask):
        # Lâp trình tại đây
        batch_size = tf.shape(q)[0]
        qw = self.wq(q)
        kw = self.wk(k)
        vw = self.wv(v)

        heads_qw = self.split_heads(qw, tf.shape(qw)[0])
        heads_kw = self.split_heads(kw, tf.shape(kw)[0])
        heads_vw = self.split_heads(vw, tf.shape(vw)[0])

        scaled = scaled_dot_product_attention(heads_qw, heads_kw, heads_vw, mask)
        tf.transpose(scaled, [0,2,1,3])

        concated= tf.reshape(scaled, (batch_size, tf.shape(qw)[1], self.d_model))

        output = self.dense(concated)
        return output

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        # Lập trình tại đây
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model,dff)
        self.layernorm1 = LayerNormalization(epsilon=rate)
        self.layernorm2 = LayerNormalization(epsilon=rate)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        # Lập trình tại đây
        q=x
        mha_out = self.mha(q, q, q, mask)
        norm_1_out = self.layernorm1(q + self.dropout1(mha_out, training=training))

        ffn_out = self.ffn(norm_1_out)
        norm_2_out = self.layernorm2(norm_1_out + self.dropout2(ffn_out, training=training))
        out2 = norm_2_out
        return out2

import numpy as np

def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    # Lập trình tại đây
    angle_rads = np.array([[get_angles(pos,i,d_model) for i in range(d_model)] for pos in range(position)])
    
    # Apply sin to even indices in the array; 2i
    angle_rads[: ,0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.expand_dims(angle_rads, axis=0)

    return tf.cast(pos_encoding, dtype=tf.float32)
