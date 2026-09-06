import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import logging

logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO
)

def build_model():
    base_model = EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(6, activation="softmax")(x)

    model = Model(base_model.input, output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():
    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        "final_dataset/train",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical"
    )

    val_data = val_gen.flow_from_directory(
        "final_dataset/val",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical"
    )

    model = build_model()

    early_stop = EarlyStopping(patience=3)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[early_stop]
    )

    acc = history.history["val_accuracy"][-1]

    logging.info(f"Training completed. Accuracy: {acc}")

    return acc, model


if __name__ == "__main__":
    train()