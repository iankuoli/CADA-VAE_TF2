
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class EncoderTemplate(tf.keras.models):
    def __init__(self, input_dim, output_dim, hidden_size_rule, latent_size):
        super(EncoderTemplate, self).__init__()

        if len(hidden_size_rule) == 2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule) == 3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]

        self.feature_encoder = Sequential()
        for i in range(len(self.layer_sizes) - 2):
            self.feature_encoder.add(layers.Dense(self.layer_sizes[i + 1],
                                                  kernel_initializer=tf.initializers.GlorotUniform))
            self.feature_encoder(tf.nn.relu())

        self._mu = layers.Dense(latent_size, kernel_initializer=tf.initializers.GlorotUniform)
        self._logvar = layers.Dense(latent_size, kernel_initializer=tf.initializers.GlorotUniform)

    def call(self, x, training=False):
        h = self.feature_encoder(x)
        mu = self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar


class DecoderTemplate(tf.keras.models):
    def __init__(self, input_dim, output_dim, hidden_size_rule):
        super(DecoderTemplate, self).__init__()

        self.layer_sizes = [input_dim, hidden_size_rule[-1], output_dim]
        self.fc1 = layers.Dense(self.layer_sizes[1], kernel_initializer=tf.initializers.GlorotUniform)
        self.act = tf.nn.relu()
        self.fc2 = layers.Dense(output_dim, kernel_initializer=tf.initializers.GlorotUniform)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
