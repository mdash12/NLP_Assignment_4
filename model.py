import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        # Create the model with a bidirectional GRU and hidden_size units
        self.basic_model = tf.keras.Sequential()
        self.basic_model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True)))
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        M = tf.tanh(rnn_outputs)
        # Compute the attention by taking the tensor dot along the common axis.
        alpha = tf.nn.softmax(tf.tensordot(M, self.omegas, [2, 0]), axis=1)
        output = tf.tanh(tf.reduce_sum(tf.multiply(alpha, rnn_outputs), axis=1))

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)  # BS * sq * es
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)  # BS * sq * es

        ### TODO(Students) START
        # ...
        # Concat the POS embedding and word embedding
        concat_emb = tf.concat([word_embed, pos_embed], axis=2) # BS * sq * (2*es)
        mask_inputs = tf.cast(inputs != 0, tf.float32)  #bs*sq

        # Use this line for passing concatenated word+pos embeddings
        # rnn_outputs = self.basic_model(concat_emb, mask=mask_inputs)

        # Use this line for passing only word embeddings
        rnn_outputs = self.basic_model(word_embed, mask=mask_inputs)
        logits = self.decoder(self.attn(rnn_outputs))  # bs * num_classes
        ### TODO(Students) END

        return {'logits': logits}

class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        self.advanced_model = tf.keras.Sequential()
        self.advanced_model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True)))
        self.advanced_model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True)))
        ### TODO(Students) END

    def call(self, inputs, pos_inputs, training):

        ### TODO(Students) START
        # ...

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)  # BS * sq * es
        input_mask = tf.cast(inputs != 0, tf.float32)  # bs*sq

        rnn_outputs = self.advanced_model(word_embed, mask=input_mask)
        max_pool_output = tf.reduce_max(rnn_outputs, axis=1)
        logits = self.decoder(max_pool_output)

        return {'logits': logits}
        ### TODO(Students END
