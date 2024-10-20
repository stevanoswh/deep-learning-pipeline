import os
import tensorflow as tf
import time  # {{ edit_1 }}: Import the time module
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
from cnnClassifier import logger

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True,
            save_weights_only=False
        )

    @property
    def _create_early_stopping_callbacks(self):
        # Add EarlyStopping callback
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Monitor validation loss
            patience=5,          # Stop training after 5 epochs with no improvement
            restore_best_weights=True  # Restore best weights after stopping
        )

    def get_tb_ckpt_callbacks(self):
        # Add EarlyStopping to the callbacks list
        return [self._create_tb_callbacks, self._create_ckpt_callbacks, self._create_early_stopping_callbacks]
