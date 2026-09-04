import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ----------------------------------------
# Settings
# ----------------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset"

# ----------------------------------------
# Data Augmentation
# ----------------------------------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("Class Labels:")
print(train_data.class_indices)

# ----------------------------------------
# MobileNetV2
# ----------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ----------------------------------------
# Classification Head
# ----------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ----------------------------------------
# Compile
# ----------------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------
# Train
# ----------------------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# ----------------------------------------
# Save Model
# ----------------------------------------

model.save(
    "models/mask_detector.keras"
)

print("Model saved successfully!")
