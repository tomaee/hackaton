import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
import pathlib

def initialModel():  #resultado usando 20 epochs!!!!!!!!      <-----------------------------------------------
    model = keras.Sequential([
        tf.keras.layers.Conv2D(128, 3, 1, activation='relu', input_shape = (64,64,3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile o modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',#pode ser substituída pela esparse_categorical_cross_entropy
                  metrics=['accuracy'])

    model.summary()

    return model

def showImage(image, label):
    # Mostre uma imagem de cada classe
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 5, 1)
    plt.xticks([])  # Remova os rótulos do eixo x
    plt.yticks([])  # Remova os rótulos do eixo y
    plt.imshow(image)
    plt.xlabel(label)

    plt.tight_layout()
    plt.show()

#def main():
#
#    image_path = "./datasets/Screenshot 2024-04-20.png"
#    image_loaded = tf.keras.utils.load_img(image_path,
#                                           target_size=(64, 64)) 
#    #show_img(image_loaded, "nuvem")
#
#    # Treine o modelo
#    model = get_cifar10_network()
#    model.fit(train_images, train_labels,batch_size=32, epochs=10,validation_data=(test_images, test_labels))
#
#    # Avalie o modelo no conjunto de teste
#    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
#
#    print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')




def trainModelAndShowResult():
    batch_size = 10
    img_height = 180
    img_width = 180
    data_dir = pathlib.Path("./datasets").with_suffix('')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width), batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width), batch_size=batch_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 2

    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
    ])


    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])



    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
    )

    image_batch, label_batch = next(iter(train_ds))

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(2):
            plt.subplot(2, 5, i + 1)
            plt.xticks([])  # Remova os rótulos do eixo x
            plt.yticks([])  # Remova os rótulos do eixo y
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.xlabel(class_names[i])
            plt.title(class_names[labels[i]])

    plt.tight_layout()
    plt.show()

    return True


def main():
    trainModelAndShowResult()


    return 1

main()

    
