
import tensorflow as tf

# export TF_FORCE_GPU_ALLOW_GROWTH=true В строку Linux

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=5000)]  # Установить лимит памяти (в МБ)
            )
        print("Memory limit set for GPUs")
    except RuntimeError as e:
        print("Error while setting memory configuration:", e)


from tensorflow.keras.models import Sequential # Linux
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalMaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
from simpson import Preprocess

# from tf_keras.models import Sequential # Для Windows
# from tf_keras.optimizers import Adam
# from tf_keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalMaxPooling2D, BatchNormalization
# from simpson import Preprocess

class Model:
    def __init__(self):
        self.img_height, self.img_width = 224, 224
        # self.batch_size = 2 Батч для 3 классов, (100+100+100)/2 = 150 на эпоху, пускал 7 эпох, точность валидации до 1
        self.batch_size = 20 # Батч для 10 классов 1000/6 = 166 по эпоху
        exp = Preprocess()
        labels = exp.labels
        self.num_classes = len(labels)  # Количество классов (персонажей)

    def layers(self):

        # Создание модели
        self.model = Sequential([
            # Сверточные слои
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            # BatchNormalization(),
            Dropout(rate=0.2),

            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            # BatchNormalization(),
            Dropout(rate=0.2),


            Conv2D(128, (3, 3), padding='same', activation='relu'),
            # BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, (3, 3), padding='same', activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            #
            # Conv2D(512, (3, 3), padding='same', activation='relu'),

            GlobalMaxPooling2D(), # Переход от 43, 43, 256 к 256
            # Выходной слой
            Dense(self.num_classes, activation='softmax')
        ])

        print(self.model.compute_output_shape(input_shape=(None, self.img_height, self.img_width, 3)))

        # Компиляция модели
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()

        return self.model