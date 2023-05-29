"""Use a deep neural network architecture with a large number of layers and parameters to explore the asymptotic behavior of neural networks."""

import tensorflow as tf
from tensorflow.keras import layers

class DeepNeuralNetwork:
    def __init__(self, num_layers, num_units):
        self.num_layers = num_layers
        self.num_units = num_units
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        for _ in range(self.num_layers):
            model.add(layers.Dense(self.num_units, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)


