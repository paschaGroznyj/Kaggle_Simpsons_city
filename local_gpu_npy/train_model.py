import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping # Ранний останов обучения, если точность валидации падает
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint # Ранний останов обучения, если точность валидации падает
from tf_keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from layers_model import Model
import os

cnn = Model()
model = cnn.layers()

class RAM:
    def __init__(self, flag_test = False):
        self.flag_test = flag_test
    def load_data(self):
        images = []
        labels = []
        with tf.device('/CPU:0'):
            for file in os.listdir("../dataset_npy"):
                if "images" in file:
                    images.append(np.load("dataset_npy/" + file))
                elif "labels" in file:
                    labels.append(np.load("dataset_npy/" + file))

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)

        return images, labels

    def data_preparation_for_test(self):
        X, y = self.load_data()  # Подгружаем изображения
        y = to_categorical(y, num_classes=cnn.num_classes)
        print(f"Shape of y: {y.shape}")  # (samples, num_classes)
        # Перемешиваем данные
        X, y = shuffle(X, y, random_state=cnn.num_classes)

        # Для раннего тестирования
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

        X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float16)

        # Печатаем форму данных для проверки
        print(f"Train shape: {X_train.shape}, {y_train.shape}")
        print(f"Test shape: {X_test.shape}, {y_test.shape}")

        return X_test, y_test


    def data_preparation_for_train(self):
        X, y = self.load_data()  # Подгружаем изображения
        y = to_categorical(y, num_classes=cnn.num_classes)
        print(f"Shape of y: {y.shape}")  # (samples, num_classes)
        # Перемешиваем данные
        self.X, self.y = shuffle(X, y, random_state=cnn.num_classes)

        # Преобразуем в тензоры
        self.X = tf.convert_to_tensor(self.X, dtype=tf.float16)
        self.y = tf.convert_to_tensor(self.y, dtype=tf.float16)


    def train_model(self):

        self.data_preparation_for_train()
        # Определяем чекпоинты
        with tf.device('/GPU:0'):
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            checkpoint = ModelCheckpoint(
                f'keras_models/best_city_simpsons_{cnn.num_classes}.keras',  # Имя файла для сохранения
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            # Обучаем модель с использованием чекпоинтов
            history = model.fit(
                self.X, self.y,
                epochs=15,
                validation_split=0.2,
                batch_size=cnn.batch_size,
                callbacks=[early_stopping, checkpoint]
            )

        hist = history.history
        x_arr = np.arange(len(hist['loss']))+1

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2 ,1)
        ax.plot(x_arr, hist['loss'], '-o', label = 'Потеря при обучении')
        ax.plot(x_arr, hist['val_loss'], '--<', label='Потеря при проверке')
        ax.legend(fontsize = 15)
        ax.set_xlabel('Эпоха', size = 15)
        ax.set_ylabel('Потеря', size=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, hist['accuracy'], '-o', label='Правильность при обучении')
        ax.plot(x_arr, hist['val_accuracy'], '--<', label='Правильность при проверке')
        ax.legend(fontsize=15)
        ax.set_xlabel('Эпоха', size=15)
        ax.set_ylabel('Правильность', size=15)

        plt.savefig('training_plot.png')


