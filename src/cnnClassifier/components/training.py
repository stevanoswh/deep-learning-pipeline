import os
import tensorflow as tf
from keras.callbacks import EarlyStopping  # Import EarlyStopping
from tensorflow.keras import layers, regularizers
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        self.model.summary()

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                fill_mode='nearest',
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                rescale=1./255
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        print(f"Number of training samples: {self.train_generator.samples}")
        print(f"Number of validation samples: {self.valid_generator.samples}")
        print(f"Number of classes: {len(self.train_generator.class_indices)}")
        print(f"Class indices: {self.train_generator.class_indices}")

        num_classes = len(self.train_generator.class_indices)
        if self.model.output_shape[-1] != num_classes:
            print(f"Adjusting model output layer to match {num_classes} classes")
            
            # Adding Dropout layer and L2 regularization
            new_output = layers.Dense(num_classes, 
                                      activation='softmax', 
                                      kernel_regularizer=regularizers.l2(0.001), 
                                      name='output_layer')(layers.Dropout(0.5)(self.model.layers[-1].output))
            self.model = tf.keras.models.Model(inputs=self.model.inputs, outputs=new_output)
            
        # Model compilation
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()

    def train(self, callback_list: list):
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        # Add early stopping to the callback list
        callback_list.append(early_stopping)

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        return history

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
