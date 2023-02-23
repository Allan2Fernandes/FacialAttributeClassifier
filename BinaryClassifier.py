import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.layers import Dense, Flatten

class BinaryClassifier:
    def __init__(self, target_size):
        self.prebuilt_model = MobileNetV3Large(
                            input_shape=target_size,
                            alpha=1.0,
                            minimalistic=False,
                            include_top=False,
                            weights="imagenet",
                            input_tensor=None,
                            classes=2,
                            pooling=None,
                            dropout_rate=0.2,
                            classifier_activation="sigmoid",
                            include_preprocessing=False,
                        )

        pass


    def build_model(self):
        self.prebuilt_model.trainable = False
        input_layer = self.prebuilt_model.input
        flatten_layer = Flatten()(self.prebuilt_model.output)
        classification_layer = Dense(units=1, activation='sigmoid')(flatten_layer)
        self.classifcation_model = tf.keras.Model(inputs = input_layer, outputs = classification_layer)
        pass

    def summarize_model(self):
        self.classifcation_model.summary()
        pass

    def compile_model(self):
        self.classifcation_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=tf.keras.metrics.BinaryAccuracy())
        pass

    def get_model(self):
        return self.classifcation_model