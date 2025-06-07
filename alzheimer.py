import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, models

img_size = (600, 600)
batch_size = 8
data_dir = "Data"

# Coleta dos caminhos das imagens e rótulos
image_paths = []
labels = []
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(class_to_idx[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

# Divisão em treino, validação e teste
X_temp, X_test, y_temp, y_test = train_test_split(
    image_paths, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42
)

def preprocess_image(path, label):
    img = load_img(path.numpy().decode(), color_mode='grayscale', target_size=img_size)
    img = img_to_array(img) / 255.0
    return img, label

def tf_preprocess(path, label):
    img, label = tf.py_function(preprocess_image, [path, label], [tf.float32, tf.int64])
    img.set_shape((img_size[0], img_size[1], 1))
    label.set_shape(())
    return img, label

def make_dataset(X, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train)
val_ds = make_dataset(X_val, y_val, shuffle=False)
test_ds = make_dataset(X_test, y_test, shuffle=False)

print("Classes:", class_names)
print("Treino:", len(X_train), "Validação:", len(X_val), "Teste:", len(X_test))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def mbconv_block(x, filters, kernel_size, strides, expand_ratio, se_ratio, drop_rate=0.2):
    input_filters = x.shape[-1]
    expanded = layers.Conv2D(input_filters * expand_ratio, 1, padding='same', use_bias=False)(x)
    expanded = layers.BatchNormalization()(expanded)
    expanded = layers.Activation('swish')(expanded)

    dw = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(expanded)
    dw = layers.BatchNormalization()(dw)
    dw = layers.Activation('swish')(dw)

    se = layers.GlobalAveragePooling2D()(dw)
    se = layers.Reshape((1, 1, se.shape[-1]))(se)
    se = layers.Conv2D(int(input_filters * se_ratio), 1, activation='swish')(se)
    se = layers.Conv2D(dw.shape[-1], 1, activation='sigmoid')(se)
    dw = layers.Multiply()([dw, se])

    pw = layers.Conv2D(filters, 1, padding='same', use_bias=False)(dw)
    pw = layers.BatchNormalization()(pw)

    if strides == 1 and input_filters == filters:
        if drop_rate:
            pw = layers.Dropout(drop_rate)(pw)
        pw = layers.Add()([x, pw])
    return pw

inputs = layers.Input(shape=(600, 600, 1))
x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)

# Blocos MBConv (ajuste conforme necessário)
x = mbconv_block(x, filters=32, kernel_size=3, strides=1, expand_ratio=1, se_ratio=0.25)
x = mbconv_block(x, filters=64, kernel_size=3, strides=2, expand_ratio=6, se_ratio=0.25)
x = mbconv_block(x, filters=128, kernel_size=3, strides=2, expand_ratio=6, se_ratio=0.25)
x = mbconv_block(x, filters=256, kernel_size=3, strides=2, expand_ratio=6, se_ratio=0.25)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Treinamento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=200,
    verbose=1,
)

# Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(test_ds)
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Perda no teste: {test_loss:.4f}")