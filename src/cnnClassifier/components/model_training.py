import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig








'''
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    # load the model from artifacts
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    
    # split the data into train and test data
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20 # 20% data for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
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
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    

    # save the model
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    # training method
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

'''



# Run on CPU (for Mac M1/M2 compatibility)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Force eager execution
tf.compat.v1.enable_eager_execution()
print("Eager execution enabled:", tf.executing_eagerly())

class Training:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        # Load the model and compile it
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("Model loaded and compiled.")

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20  # 20% for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="sparse"  # Use sparse labels (integers)
        )

        # Validation generator
        valid_datagenerator = ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator with optional augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Convert the generators to tf.data.Datasets.
        # We use a generator expression to yield (x, y) batches.
        self.train_dataset = tf.data.Dataset.from_generator(
            lambda: ((x, y) for x, y in self.train_generator),
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.params_image_size), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        self.valid_dataset = tf.data.Dataset.from_generator(
            lambda: ((x, y) for x, y in self.valid_generator),
            output_signature=(
                tf.TensorSpec(shape=(None, *self.config.params_image_size), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )

        # Map over the dataset to ensure any RaggedTensors are converted to dense.
        def to_dense(x, y):
            if isinstance(x, tf.RaggedTensor):
                x = x.to_tensor()
            return x, y

        self.train_dataset = self.train_dataset.map(to_dense)
        self.valid_dataset = self.valid_dataset.map(to_dense)

        print("Data generators converted to tf.data.Datasets.")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        print(f"Model saved at {path}")

    def train(self):
        # Compute steps from the original generator properties
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        print("Starting model training...")

        history = self.model.fit(
            self.train_dataset,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_dataset,
            validation_steps=self.validation_steps
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print("Model training completed.")
        return history



