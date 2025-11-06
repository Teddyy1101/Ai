import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Enable mixed precision for better performance on GPUs
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print(f'Mixed precision enabled. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}')
except:
    print('Mixed precision not available')

# === Path Configuration ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "mouth")
MODEL_OUT = os.path.join(BASE_DIR, "mouth_model.h5")
CLASS_INDICES_OUT = os.path.join(BASE_DIR, "class_indices.json")

# === Training Configuration ===
IMG_SIZE = (160, 160)  # Increased size for better feature extraction
BATCH_SIZE = 32
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 20
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
AUTOTUNE = tf.data.AUTOTUNE

# === Data Augmentation ===
def augment(image, label):
    """Apply augmentations to the image"""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random rotation
    image = tf.image.rotate(
        image, 
        tf.random.uniform(shape=[], minval=-0.1, maxval=0.1) * np.pi
    )
    
    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random zoom and crop
    image = tf.image.random_zoom(image, [0.9, 1.1])
    image = tf.image.resize(image, IMG_SIZE)
    
    # Ensure pixel values are still in [0,1]
    image = tf.clip_by_value(image, 0, 1)
    
    return image, label

# === Data Loading and Preprocessing ===
print("\n🔍 Loading and preprocessing data...")

# Calculate class weights for imbalanced datasets
def calculate_class_weights():
    class_counts = {}
    for root, dirs, files in os.walk(DATA_DIR):
        if os.path.basename(root) in ['yawn', 'normal']:
            class_name = os.path.basename(root)
            class_counts[class_name] = len(files)
    
    total = sum(class_counts.values())
    class_weights = {i: total / (len(class_counts) * count) 
                    for i, (_, count) in enumerate(sorted(class_counts.items()))}
    return class_weights

# Data augmentation and preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Data loaders
train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class indices and weights
class_indices = train_gen.class_indices
num_classes = len(class_indices)
class_weights = calculate_class_weights()

print(f"\n📊 Dataset Summary:")
print(f"- Training samples: {train_gen.samples}")
print(f"- Validation samples: {val_gen.samples}")
print(f"- Number of classes: {num_classes}")
print(f"- Class indices: {class_indices}")
print(f"- Class weights: {class_weights}")

# === Model Architecture ===
def build_model(input_shape, num_classes):
    """Build and compile the model with EfficientNetB0 as base"""
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomZoom(0.1)(x)
    
    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    
    # Additional layers
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    
    return model, base_model

# Build and compile the model
print("\n🏗️  Building model...")
model, base_model = build_model(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)

# Custom learning rate schedule
initial_learning_rate = INITIAL_LR
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# === Callbacks ===
checkpoint = callbacks.ModelCheckpoint(
    MODEL_OUT,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard callback
tensorboard_callback = callbacks.TensorBoard(
    log_dir=os.path.join(BASE_DIR, 'logs'),
    histogram_freq=1
)

callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard_callback]

# === Training Phase 1: Train the top layers ===
print("\n🚀 Starting Phase 1: Training top layers...")
history = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights
)

# === Fine-tuning Phase: Unfreeze some layers of the base model ===
print("\n🔧 Starting Phase 2: Fine-tuning...")

# Unfreeze the top layers of the base model
base_model.trainable = True

# Freeze the bottom layers and only train the top ones
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile the model for fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
fine_tune_epochs = FINE_TUNE_EPOCHS
total_epochs = INITIAL_EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_gen,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights
)

# === Save the model and class indices ===
print("\n💾 Saving model and class indices...")
model.save(MODEL_OUT)
with open(CLASS_INDICES_OUT, 'w') as f:
    json.dump(class_indices, f)

print(f"\n✅ Training complete! Model saved to {MODEL_OUT}")
print(f"   Class indices saved to {CLASS_INDICES_OUT}")

# === Evaluate the model ===
print("\n📊 Evaluating model...")
result = model.evaluate(val_gen)
print(f"\nEvaluation results:")
for name, value in zip(model.metrics_names, result):
    print(f"{name}: {value:.4f}")

# Generate predictions for confusion matrix
y_pred = model.predict(val_gen)
y_pred = np.argmax(y_pred, axis=1)
y_true = val_gen.classes

# Calculate and print classification report
from sklearn.metrics import classification_report, confusion_matrix
print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_indices.keys()))

# Plot training history
def plot_training_history(history):
    import matplotlib.pyplot as plt
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.show()

# Plot training history
plot_training_history(history)
print("\n🎉 Training completed successfully!")

# === (Tùy chọn) Fine-tune thêm tầng MobileNetV2 ===
print("\n🔧 Fine-tuning phase (unfreeze last 30 layers)...")
base_model = model.layers[1]  # lấy base model
for layer in base_model.layers[-30:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# === Lưu model và class indices ===
model.save(MODEL_OUT)
with open(CLASS_INDICES_OUT, "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ Model saved to:", MODEL_OUT)
print("✅ Class indices saved to:", CLASS_INDICES_OUT)
