import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, regularizers, applications
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier import logger

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        # Load the base ResNet50 model without the top fully connected layers
        self.model = applications.ResNet50(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=False
        )
        logger.info(f"Base model loaded: ResNet50")
        self.save_model(path=self.config.base_model_path, model=self.model)
        logger.info(f"Base model saved at: {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add custom layers
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(classes, activation="softmax", kernel_regularizer=regularizers.l2(0.001))(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        logger.info("Full model prepared and compiled")
        full_model.summary()
        return full_model

    def update_base_model(self):
        if self.model is None:
            logger.info("Base model not found. Loading base model.")
            self.get_base_model()

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        logger.info(f"Updated model saved at: {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def load_model(self, path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)