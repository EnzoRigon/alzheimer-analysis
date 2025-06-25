# Importações de bibliotecas necessárias
from matplotlib.pyplot import imshow
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input, Dropout

# Configurações iniciais
img_size = (32, 32)  # Tamanho para redimensionar as imagens
batch_size = 8
data_dir = "Data"    # Diretório com as imagens

# Preparação das listas de caminhos de imagens e rótulos
image_paths = []
labels = []
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
print(class_to_idx)

# Percorre as pastas de cada classe e armazena caminhos e rótulos
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(class_to_idx[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

# Divide os dados em treino, validação e teste (estratificado)
X_temp, X_test, y_temp, y_test = train_test_split(
    image_paths, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42
)

# Função para carregar e pré-processar uma imagem
def preprocess_image(path, label):
    img = load_img(path.numpy().decode(), color_mode='grayscale', target_size=img_size)
    img = img_to_array(img) / 255.0
    return img, label

# Função wrapper para usar no tf.data
def tf_preprocess(path, label):
    img, label = tf.py_function(preprocess_image, [path, label], [tf.float32, tf.int64])
    img.set_shape((img_size[0], img_size[1], 1))
    label.set_shape(())
    return img, label

# Função para criar datasets do TensorFlow
def make_dataset(X, y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Criação dos datasets de treino, validação e teste
train_ds = make_dataset(X_train, y_train)
val_ds = make_dataset(X_val, y_val, shuffle=False)
test_ds = make_dataset(X_test, y_test, shuffle=False)

print("Classes:", class_names)
print("Treino:", len(X_train), "Validação:", len(X_val), "Teste:", len(X_test))

# Definição do modelo sequencial (Arquitetura 1)
model = Sequential()

# Camada de entrada
model.add(Input(shape=(img_size[0], img_size[1], 1)))
# Primeira camada convolucional + pooling + dropout
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Segunda camada convolucional + pooling + dropout
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Achata os mapas de ativação para vetor
model.add(Flatten())
# Camada densa intermediária
model.add(Dense(64, activation='relu'))
# Camada de saída com softmax para classificação multiclasse
model.add(Dense(4, activation='softmax'))

# Compilação do modelo com Adamax e função de perda apropriada
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

print(model.summary())

# Early stopping para evitar overfitting
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Treinamento do modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1,
    callbacks=[early_stop]
)

# Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(test_ds)
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Perda no teste: {test_loss:.4f}")

# Função para prever e mostrar uma imagem
def predict(path):
    print(f'Predicao de {path}')
    img = load_img(path, color_mode='grayscale', target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    res = model.predict_on_batch(img_array)
    classification = np.where(res == np.amax(res))[1][0]
    print(res)
    print(np.amax(res))
    print(classification)

    imshow(img)
    print(str(res[0][classification] * 100) + '% Confidence This Is ' + class_names[classification])

    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title(f"Classe prevista: {class_names[classification]}")
    plt.axis('off')
    plt.show()

# Exemplos de predição em imagens específicas de cada classe
predict(f"{data_dir}/Moderate Dementia/OAS1_0308_MR1_mpr-1_101.jpg")
predict(f"{data_dir}/Very mild Dementia/OAS1_0380_MR1_mpr-4_158.jpg")
predict(f"{data_dir}/Mild Dementia/OAS1_0028_MR1_mpr-1_145.jpg")
predict(f"{data_dir}/Non Demented/OAS1_0381_MR1_mpr-1_100.jpg")